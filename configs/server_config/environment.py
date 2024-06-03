ENV_PATH = "/root/enter/envs/parrotbot/bin"
DEBUG = True
DATABASE_SELECTION = "mysql"
DATABASE_PREX = 's-agent_'
MAX_STREAM_WORKER = 20


# Support 2 DBs
def db_selection():
    if DATABASE_SELECTION == 'postgre':
        from configs.server_config.postgre_config import BASES
    elif DATABASE_SELECTION == 'mysql':
        from configs.server_config.mysql_config import BASES
    else:
        raise ValueError("Unsupported database selection.")
    return BASES
