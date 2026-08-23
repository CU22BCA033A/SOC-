import time

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.agent.router import AgentError, generate_general_answer, generate_grounded_answer
from app.agent.smalltalk import smalltalk_reply
from app.config import get_settings
from app.db import get_db
from app.models import Conversation, Message, RetrievalAuditLog
from app.rag.vectorstore import get_vector_store
from app.schemas import ChatRequest, ChatResponse, Citation

router = APIRouter()

MAX_HISTORY_TURNS = 10
NO_MATCH_MESSAGE = (
    "I couldn't find anything in our knowledge base about that, so I don't want to "
    "guess. I can hand this off to a human agent who can help — would you like me to?"
)


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    settings = get_settings()
    started = time.monotonic()

    conversation = _get_or_create_conversation(db, req.conversation_id, req.session_id)
    history = _load_history(db, conversation.id)

    escalated = False
    escalation_reason: str | None = None
    retrieved = []
    top_score = 0.0
    grounded = False  # only True when `answer` actually cites `retrieved`

    canned = smalltalk_reply(req.message)
    if canned is not None:
        # Plain greeting/thanks/farewell — not a policy question, so skip
        # retrieval entirely rather than let it fall through the confidence
        # gate and escalate on a bare "hi".
        answer = canned
    else:
        store = get_vector_store()
        retrieved = store.query(req.message, top_k=settings.retrieval_top_k)
        top_score = retrieved[0].score if retrieved else 0.0

        if not retrieved or top_score < settings.retrieval_confidence_threshold:
            # Low/no retrieval match doesn't necessarily mean "out of scope" --
            # it could be general conversation the KB was never going to cover.
            # One LLM call both decides and answers: it either responds
            # normally (general chat, no citations) or signals "this needs
            # grounding I don't have" by returning None, in which case we
            # escalate rather than risk an invented policy answer.
            try:
                general_answer = generate_general_answer(req.message, history)
            except AgentError as e:
                answer = (
                    f"{e} I've noted this so a human agent can follow up — sorry for the trouble."
                )
                escalated = True
                escalation_reason = "agent_error"
            else:
                if general_answer is None:
                    answer = NO_MATCH_MESSAGE
                    escalated = True
                    escalation_reason = "low_retrieval_confidence"
                else:
                    answer = general_answer
        else:
            chunk_dicts = [{"doc": r.doc, "heading": r.heading, "text": r.text} for r in retrieved]
            try:
                answer = generate_grounded_answer(req.message, history, chunk_dicts)
                grounded = True
            except AgentError as e:
                answer = (
                    f"{e} I've noted this so a human agent can follow up — sorry for the trouble."
                )
                escalated = True
                escalation_reason = "agent_error"

    latency_ms = int((time.monotonic() - started) * 1000)

    db.add(Message(conversation_id=conversation.id, role="user", content=req.message))
    db.add(Message(conversation_id=conversation.id, role="assistant", content=answer))
    db.add(
        RetrievalAuditLog(
            conversation_id=conversation.id,
            query=req.message,
            retrieved_chunks=[
                {"doc": r.doc, "heading": r.heading, "score": r.score} for r in retrieved
            ],
            top_score=top_score,
            escalated=escalated,
            escalation_reason=escalation_reason,
            answer=answer,
            latency_ms=latency_ms,
        )
    )
    db.commit()

    citations = (
        [Citation(doc=r.doc, heading=r.heading, score=r.score) for r in retrieved]
        if grounded
        else []
    )

    return ChatResponse(
        conversation_id=conversation.id,
        answer=answer,
        citations=citations,
        escalated=escalated,
        escalation_reason=escalation_reason,
        latency_ms=latency_ms,
    )


def _get_or_create_conversation(db: Session, conversation_id: str | None, session_id: str) -> Conversation:
    if conversation_id:
        existing = db.get(Conversation, conversation_id)
        if existing:
            return existing
    conversation = Conversation(session_id=session_id)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


def _load_history(db: Session, conversation_id: str) -> list[dict[str, str]]:
    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
        .all()
    )
    recent = messages[-MAX_HISTORY_TURNS * 2 :]
    return [{"role": m.role, "content": m.content} for m in recent]
