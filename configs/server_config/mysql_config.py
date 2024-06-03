from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session
from sqlalchemy_utils import database_exists, create_database

MYSQL_SIGNATURE_1 = 's-agent'
MYSQL_SETTINGS = {
    MYSQL_SIGNATURE_1: {
        "HOST": 'localhost',
        "PORT": 19779,
        "USERNAME": "root",
        "PASSWORD": "ParrotBot236",
        "DB": "agent",
        "CONNECTOR": "pymysql"
    }
}


#  ----------------------- System functions ------------------------------ #
def get_db_uri(CONNECTOR, USERNAME, PASSWORD, HOST, PORT, DB):
    return f'mysql+{CONNECTOR}://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DB}'


def get_db_session_sql(db) -> Session:
    return sessionmaker(ENGINES[db], autocommit=False, autoflush=False)()


def auto_create_db(engine_name=MYSQL_SIGNATURE_1):
    if not database_exists(ENGINES[engine_name].url):
        create_database(ENGINES[engine_name].url)
        print(f"Database Created: {database_exists(ENGINES[engine_name].url)}")


#  ----------------------- System functions ------------------------------ #

# Engine AND BASE setting, can add as many as wanted
DB_URI_1 = get_db_uri(**MYSQL_SETTINGS[MYSQL_SIGNATURE_1])

ENGINES = {
    MYSQL_SIGNATURE_1: create_engine(
        DB_URI_1,
        max_overflow=-1,
        pool_pre_ping=True,
        pool_recycle=3600)
}

auto_create_db()

BASES = {
    MYSQL_SIGNATURE_1: declarative_base(ENGINES[MYSQL_SIGNATURE_1])
}
