from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Enum, Float, Text, Boolean, CheckConstraint
from sqlalchemy import UniqueConstraint
from configs.server_config.environment import db_selection, DATABASE_PREX
from configs.server_config.mysql_config import MYSQL_SIGNATURE_1
import enum

Base = db_selection()[MYSQL_SIGNATURE_1]


class SystemCategory(enum.Enum):
    HUMAN = 1
    ROBOT = 2


class LLMRequestType(enum.Enum):
    API = 1
    LOCAL = 2


class MessageType(enum.Enum):
    system = 1
    self = 2
    exterior = 3


# =================================== Human Level ============================= #
class User(Base):
    __tablename__ = DATABASE_PREX + "users"

    id = Column(Integer, autoincrement=True, primary_key=True)
    name = Column(String(30), nullable=False)
    system_type = Column(Enum(SystemCategory), default=1)
    create_time = Column(DateTime)
    last_update_time = Column(DateTime)


# =================================== Agent level ============================= #
class Agents(Base):
    __tablename__ = DATABASE_PREX + "agents"
    # __table_args__ = (
    #     {"extend_existing":True,'mysql_charset': 'utf8',}
    # )
    # __table_args__ = (
    #     UniqueConstraint("user_id"),
    # )

    id = Column(Integer, autoincrement=True, primary_key=True)
    name = Column(String(30), nullable=False)
    description = Column(Text, nullable=False)
    system_type = Column(Enum(SystemCategory), default=2)
    task_complete = Column(Integer, default=0)
    task_success = Column(Integer, default=0)
    create_time = Column(DateTime)
    last_update_time = Column(DateTime)


class AgentPlanningState(Base):
    __tablename__ = DATABASE_PREX + "agent_planning_state"

    id = Column(Integer, autoincrement=True, primary_key=True)
    state_name = Column(String(30), nullable=False)
    state_description = Column(Text, nullable=False)
    create_time = Column(DateTime)
    last_update_time = Column(DateTime)


class AgentTrajectory(Base):
    __tablename__ = DATABASE_PREX + "agent_trajectory"

    id = Column(Integer, autoincrement=True, primary_key=True)
    agent_id = Column(Integer, nullable=False)  # Foreign key of the agents
    planning_state_id = Column(Integer, nullable=False)  # Foreign key of the agent_planning_state
    trajectory_text = Column(Text, nullable=False)
    token_length = Column(Integer)
    create_time = Column(DateTime)
    reward = Column(Integer)


class Tools(Base):
    __tablename__ = DATABASE_PREX + "tools"

    id = Column(Integer, autoincrement=True, primary_key=True)
    name = Column(String(30), nullable=False)
    description = Column(Text, nullable=False)
    tool_path = Column(Text, nullable=False)
    tool_callback_methods = Column(Text, nullable=False)
    create_time = Column(DateTime)
    last_update_time = Column(DateTime)


class AgentTools(Base):
    __tablename__ = DATABASE_PREX + "agent_tools"

    id = Column(Integer, autoincrement=True, primary_key=True)
    agent_id = Column(Integer, nullable=False)  # Foreign key of the agents
    tool_id = Column(Integer, nullable=False)  # Foreign key of the tools
    create_time = Column(DateTime)
    success_state = Column(Boolean)


class AgentMessages(Base):
    __tablename__ = DATABASE_PREX + "agent_messages"

    id = Column(Integer, autoincrement=True, primary_key=True)
    message_content = Column(Text)
    agent_id = Column(Integer, nullable=False)  # Foreign key of the agents
    message_type = Column(Enum(MessageType))
    create_time = Column(DateTime)


# =================================== Multi-Agent level ============================= #
class Teams(Base):
    __tablename__ = DATABASE_PREX + "teams"

    id = Column(Integer, autoincrement=True, primary_key=True)
    name = Column(String(30), nullable=False)
    description = Column(Text, nullable=False)
    task_complete = Column(Integer, default=0)
    task_success = Column(Integer, default=0)
    create_time = Column(DateTime)
    last_update_time = Column(DateTime)


class Groups(Base):
    __tablename__ = DATABASE_PREX + "groups"

    id = Column(Integer, autoincrement=True, primary_key=True)
    name = Column(String(30), nullable=False)
    description = Column(Text, nullable=False)
    task_complete = Column(Integer, default=0)
    task_success = Column(Integer, default=0)
    superior_group_id = Column(Integer, nullable=True)  # Foreign key of self
    collab_group_id = Column(Integer, nullable=True)  # Foreign key of self
    create_time = Column(DateTime)
    last_update_time = Column(DateTime)


class AgentGroupFlow(Base):
    __tablename__ = DATABASE_PREX + "agent_group_flow"

    id = Column(Integer, autoincrement=True, primary_key=True)
    from_team_id = Column(Integer, nullable=False)  # Foreign key of team
    from_group_id = Column(Integer, nullable=False)  # Foreign key of group
    from_agent_id = Column(Integer, nullable=False)  # Foreign key of agent
    to_team_id = Column(Integer, nullable=False)  # Foreign key of team
    to_group_id = Column(Integer, nullable=False)  # Foreign key of group
    to_agent_id = Column(Integer, nullable=False)  # Foreign key of agent


class AgentTeamGroup(Base):
    __tablename__ = DATABASE_PREX + "agent_team_group"

    id = Column(Integer, autoincrement=True, primary_key=True)
    team_id = Column(Integer, nullable=False)  # Foreign key of team
    group_id = Column(Integer, nullable=False)  # Foreign key of group
    agent_id = Column(Integer, nullable=False)  # Foreign key of agent
    create_time = Column(DateTime)

    __table_args__ = (
        CheckConstraint('team_id != group_id OR team_id != agent_id OR group_id != agent_id', name='check_ids_diff'),
    )


class UserTeamGroup(Base):
    __tablename__ = DATABASE_PREX + "user_team_group"

    id = Column(Integer, autoincrement=True, primary_key=True)
    team_id = Column(Integer, nullable=False)  # Foreign key of team
    group_id = Column(Integer, nullable=False)  # Foreign key of group
    user_id = Column(Integer, nullable=False)  # Foreign key of agent
    create_time = Column(DateTime)

    __table_args__ = (
        CheckConstraint('team_id != group_id OR team_id != agent_id OR group_id != usr_id', name='check_ids_diff'),
    )


class Task(Base):
    __tablename__ = DATABASE_PREX + "tasks"

    id = Column(Integer, autoincrement=True, primary_key=True)
    name = Column(Text, nullable=False)
    description = Column(Text, nullable=False)
    create_time = Column(DateTime)
    last_update_time = Column(DateTime)


class Chats(Base):
    __tablename__ = DATABASE_PREX + "chats"

    id = Column(Integer, autoincrement=True, primary_key=True)
    team_id = Column(Integer, nullable=False)  # Foreign key of team
    group_id = Column(Integer, nullable=False)  # Foreign key of group
    chat_content = Column(Text)
    send_source_id = Column(Integer)
    send_source_category = Column(Enum(SystemCategory))
    send_time = Column(DateTime)


class ChatAt(Base):
    __tablename__ = DATABASE_PREX + "chat_at"

    id = Column(Integer, autoincrement=True, primary_key=True)
    chat_id = Column(Integer)  # Foreign key of chat
    at_target_id = Column(Integer)  # Foreign key of chat
    at_target_category = Column(Enum(SystemCategory))


# =================================== LLM Management system ========================= #
class LLMSource(Base):
    __tablename__ = DATABASE_PREX + "LLM_source"

    id = Column(Integer, autoincrement=True, primary_key=True)
    name = Column(Text, nullable=False)
    description = Column(Text, nullable=False)
    request_type = Column(Enum(LLMRequestType), nullable=False)
    call_class = Column(Enum(LLMRequestType), nullable=False)
    api_key = Column(Enum(LLMRequestType), nullable=True)
    create_time = Column(DateTime)
    last_update_time = Column(DateTime)
