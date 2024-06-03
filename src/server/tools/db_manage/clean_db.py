# from configs.server_config.environment import DATABASE_SELECTION
# if DATABASE_SELECTION == "postgre":
#     from configs.server_config.postgre_config import get_db_session
# elif DATABASE_SELECTION == "mysql":
#     from configs.server_config.mysql_config import get_db_session_sql, MYSQL_SETTINGS, MYSQL_SIGNATURE_1
# from sqlalchemy.sql import text
#
# if DATABASE_SELECTION == "postgre":
#     with get_db_session(MYSQL_SIGNATURE_1) as session:
#         session.execute(text("""
#         DO $$ DECLARE
#           r RECORD;
#         BEGIN
#           FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = current_schema()) LOOP
#             EXECUTE 'DROP TABLE ' || quote_ident(r.tablename) || ' CASCADE';
#           END LOOP;
#         END $$;
#       """))
#         try:
#             session.commit()
#         except Exception as e:
#             session.rollback()
#             raise e
# elif DATABASE_SELECTION == "mysql":
#     with get_db_session_sql(MYSQL_SIGNATURE_1) as session:
#         session.execute(text(f"""
#         SET foreign_key_checks = 0;
#         """))
#         session.execute(text(f"""
#         SELECT CONCAT('DROP TABLE ', TABLE_NAME, ';')
#         FROM INFORMATION_SCHEMA.tables
#         WHERE TABLE_SCHEMA = '{MYSQL_SETTINGS[MYSQL_SIGNATURE_1]['DB']}';
#       """))
#         session.execute(text("""
#         SET foreign_key_checks = 1;
#         """))
#         try:
#             session.commit()
#             print('All table dropped.')
#         except Exception as e:
#             session.rollback()
#             raise e

from src.server.blueprints.agents.models import Base as A_BASE

bases = [A_BASE]

for base in bases:
    base.metadata.drop_all()
    print('All table dropped.')
    base.metadata.create_all()
    print('All table recreated.')