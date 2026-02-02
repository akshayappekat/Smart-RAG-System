"""Conversation memory management for multi-turn interactions."""

import asyncio
import json
import sqlite3
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

import logging
logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Single turn in a conversation."""
    turn_id: str
    session_id: str
    user_message: str
    assistant_response: str
    timestamp: datetime
    confidence: float
    sources_used: List[str]
    execution_time: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ConversationSession:
    """Complete conversation session."""
    session_id: str
    user_id: Optional[str]
    start_time: datetime
    last_activity: datetime
    turns: List[ConversationTurn]
    context: Dict[str, Any]
    is_active: bool = True

    def __post_init__(self):
        if not self.turns:
            self.turns = []
        if not self.context:
            self.context = {}


class ConversationMemory:
    """Manages conversation history and context."""
    
    def __init__(self, db_path: str = "data/conversations.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for active sessions
        self.active_sessions: Dict[str, ConversationSession] = {}
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Conversation memory initialized with database: {self.db_path}")
    
    def _init_database(self):
        """Initialize SQLite database for conversation storage."""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    start_time TEXT,
                    last_activity TEXT,
                    context TEXT,
                    is_active INTEGER
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS turns (
                    turn_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    user_message TEXT,
                    assistant_response TEXT,
                    timestamp TEXT,
                    confidence REAL,
                    sources_used TEXT,
                    execution_time REAL,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            """)
            
            conn.commit()
    
    async def create_session(self, user_id: Optional[str] = None, 
                           initial_context: Dict[str, Any] = None) -> str:
        """Create new conversation session."""
        
        session_id = f"session_{int(time.time() * 1000)}"
        current_time = datetime.now()
        
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            start_time=current_time,
            last_activity=current_time,
            turns=[],
            context=initial_context or {},
            is_active=True
        )
        
        # Store in memory cache
        self.active_sessions[session_id] = session
        
        # Store in database
        await self._save_session_to_db(session)
        
        logger.info(f"Created new conversation session: {session_id}")
        return session_id
    
    async def add_turn(self, session_id: str, user_message: str, 
                      assistant_response: str, confidence: float,
                      sources_used: List[str], execution_time: float,
                      metadata: Dict[str, Any] = None) -> str:
        """Add new turn to conversation."""
        
        turn_id = f"turn_{session_id}_{int(time.time() * 1000)}"
        current_time = datetime.now()
        
        turn = ConversationTurn(
            turn_id=turn_id,
            session_id=session_id,
            user_message=user_message,
            assistant_response=assistant_response,
            timestamp=current_time,
            confidence=confidence,
            sources_used=sources_used,
            execution_time=execution_time,
            metadata=metadata or {}
        )
        
        # Update session
        session = await self.get_session(session_id)
        if session:
            session.turns.append(turn)
            session.last_activity = current_time
            
            # Update memory cache
            self.active_sessions[session_id] = session
            
            # Save to database
            await self._save_turn_to_db(turn)
            await self._update_session_in_db(session)
        
        logger.debug(f"Added turn to session {session_id}: {turn_id}")
        return turn_id
    
    async def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get conversation session by ID."""
        
        # Check memory cache first
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Load from database
        session = await self._load_session_from_db(session_id)
        if session:
            self.active_sessions[session_id] = session
        
        return session
    
    async def get_conversation_context(self, session_id: str, 
                                     max_turns: int = 10) -> Dict[str, Any]:
        """Get conversation context for the session."""
        
        session = await self.get_session(session_id)
        if not session:
            return {}
        
        # Get recent turns
        recent_turns = session.turns[-max_turns:] if session.turns else []
        
        # Build context
        context = {
            "session_id": session_id,
            "conversation_history": [],
            "session_context": session.context,
            "turn_count": len(session.turns),
            "session_duration": (
                (session.last_activity - session.start_time).total_seconds()
                if session.turns else 0
            )
        }
        
        # Add conversation history
        for turn in recent_turns:
            context["conversation_history"].append({
                "user": turn.user_message,
                "assistant": turn.assistant_response,
                "timestamp": turn.timestamp.isoformat(),
                "confidence": turn.confidence
            })
        
        return context
    
    async def update_session_context(self, session_id: str, 
                                   context_updates: Dict[str, Any]):
        """Update session context."""
        
        session = await self.get_session(session_id)
        if session:
            session.context.update(context_updates)
            session.last_activity = datetime.now()
            
            # Update cache and database
            self.active_sessions[session_id] = session
            await self._update_session_in_db(session)
    
    async def end_session(self, session_id: str):
        """End conversation session."""
        
        session = await self.get_session(session_id)
        if session:
            session.is_active = False
            session.last_activity = datetime.now()
            
            # Update database
            await self._update_session_in_db(session)
            
            # Remove from active cache
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
        
        logger.info(f"Ended conversation session: {session_id}")
    
    async def cleanup_old_sessions(self, max_age_days: int = 30):
        """Clean up old inactive sessions."""
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        with sqlite3.connect(self.db_path) as conn:
            # Delete old turns
            conn.execute("""
                DELETE FROM turns 
                WHERE session_id IN (
                    SELECT session_id FROM sessions 
                    WHERE last_activity < ? AND is_active = 0
                )
            """, (cutoff_date.isoformat(),))
            
            # Delete old sessions
            conn.execute("""
                DELETE FROM sessions 
                WHERE last_activity < ? AND is_active = 0
            """, (cutoff_date.isoformat(),))
            
            conn.commit()
        
        logger.info(f"Cleaned up sessions older than {max_age_days} days")
    
    async def get_session_stats(self) -> Dict[str, Any]:
        """Get conversation memory statistics."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total sessions
            cursor.execute("SELECT COUNT(*) FROM sessions")
            total_sessions = cursor.fetchone()[0]
            
            # Active sessions
            cursor.execute("SELECT COUNT(*) FROM sessions WHERE is_active = 1")
            active_sessions = cursor.fetchone()[0]
            
            # Total turns
            cursor.execute("SELECT COUNT(*) FROM turns")
            total_turns = cursor.fetchone()[0]
            
            # Average turns per session
            cursor.execute("""
                SELECT AVG(turn_count) FROM (
                    SELECT COUNT(*) as turn_count 
                    FROM turns 
                    GROUP BY session_id
                )
            """)
            avg_turns = cursor.fetchone()[0] or 0
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "cached_sessions": len(self.active_sessions),
            "total_turns": total_turns,
            "average_turns_per_session": round(avg_turns, 2)
        }
    
    async def _save_session_to_db(self, session: ConversationSession):
        """Save session to database."""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sessions 
                (session_id, user_id, start_time, last_activity, context, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.user_id,
                session.start_time.isoformat(),
                session.last_activity.isoformat(),
                json.dumps(session.context),
                1 if session.is_active else 0
            ))
            conn.commit()
    
    async def _save_turn_to_db(self, turn: ConversationTurn):
        """Save turn to database."""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO turns 
                (turn_id, session_id, user_message, assistant_response, 
                 timestamp, confidence, sources_used, execution_time, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                turn.turn_id,
                turn.session_id,
                turn.user_message,
                turn.assistant_response,
                turn.timestamp.isoformat(),
                turn.confidence,
                json.dumps(turn.sources_used),
                turn.execution_time,
                json.dumps(turn.metadata)
            ))
            conn.commit()
    
    async def _update_session_in_db(self, session: ConversationSession):
        """Update session in database."""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE sessions 
                SET last_activity = ?, context = ?, is_active = ?
                WHERE session_id = ?
            """, (
                session.last_activity.isoformat(),
                json.dumps(session.context),
                1 if session.is_active else 0,
                session.session_id
            ))
            conn.commit()
    
    async def _load_session_from_db(self, session_id: str) -> Optional[ConversationSession]:
        """Load session from database."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Load session
            cursor.execute("""
                SELECT user_id, start_time, last_activity, context, is_active
                FROM sessions WHERE session_id = ?
            """, (session_id,))
            
            session_row = cursor.fetchone()
            if not session_row:
                return None
            
            # Load turns
            cursor.execute("""
                SELECT turn_id, user_message, assistant_response, timestamp,
                       confidence, sources_used, execution_time, metadata
                FROM turns WHERE session_id = ?
                ORDER BY timestamp
            """, (session_id,))
            
            turn_rows = cursor.fetchall()
            
            # Build session object
            turns = []
            for row in turn_rows:
                turn = ConversationTurn(
                    turn_id=row[0],
                    session_id=session_id,
                    user_message=row[1],
                    assistant_response=row[2],
                    timestamp=datetime.fromisoformat(row[3]),
                    confidence=row[4],
                    sources_used=json.loads(row[5]),
                    execution_time=row[6],
                    metadata=json.loads(row[7])
                )
                turns.append(turn)
            
            session = ConversationSession(
                session_id=session_id,
                user_id=session_row[0],
                start_time=datetime.fromisoformat(session_row[1]),
                last_activity=datetime.fromisoformat(session_row[2]),
                turns=turns,
                context=json.loads(session_row[3]),
                is_active=bool(session_row[4])
            )
            
            return session


# Global conversation memory instance
conversation_memory = ConversationMemory()