"""Configuration for database connection."""
from pydantic import BaseModel, Field
import mysql.connector

class DBConfig(BaseModel):
    """Database connection configuration."""
    db_type: str = Field(..., description="Database type, e.g. sqlite, mysql, etc.")
    db_name: str = Field(..., description="Database name.")
    file_path: str = Field("", description="File path for file-based database.")
    db_host: str = Field("", description="Database host.")
    db_port: int = Field(0, description="Database port.")
    db_user: str = Field("", description="Database user.")
    db_pwd: str = Field("", description="Database password.")
    comment: str = Field("", description="Comment for the database.")

    def get_table_structure(config):
        # connect to the MySQL database
        conn = mysql.connector.connect(
            host=config.db_host,
            port=config.db_port,
            user=config.db_user,
            password=config.db_pwd,
            database=config.db_name
        )

        cursor = conn.cursor(dictionary=True)

        # Execute SQL query
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        table_structure = {}

        for table in tables:
            table_name = table['Tables_in_' + config.db_name]
            # Query to get columns with comments
            cursor.execute(f"""
                SELECT COLUMN_NAME, COLUMN_TYPE, COLUMN_COMMENT
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = '{config.db_name}' AND TABLE_NAME = '{table_name}'
            """)
            columns = cursor.fetchall()

            table_structure[table_name] = [
                {'Field': column['COLUMN_NAME'], 'Type': column['COLUMN_TYPE'], 'Comment': column['COLUMN_COMMENT']}
                for column in columns
            ]

        # Close cursor and connection
        cursor.close()
        conn.close()

        return table_structure

    def get_table_data(config):
        # connect to the MySQL database
        conn = mysql.connector.connect(
            host=config.db_host,
            port=config.db_port,
            user=config.db_user,
            password=config.db_pwd,
            database=config.db_name
        )

        cursor = conn.cursor(dictionary=True)

        # Execute SQL query
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()

        table_structure = {}

        for table in tables:
            table_name = table['Tables_in_' + config.db_name]

            # Query to get columns with comments
            cursor.execute(f"""
                        SELECT *
                        FROM {table_name} limit 1;
                    """)
            columns = cursor.fetchall()
            table_structure[table_name] = [column for column in columns]
        # Close cursor and connection
        cursor.close()
        conn.close()
        return table_structure

    def get_foreign_keys(config):
        # Connect to the MySQL database
        conn = mysql.connector.connect(
            host=config.db_host,
            port=config.db_port,
            user=config.db_user,
            password=config.db_pwd,
            database=config.db_name
        )

        cursor = conn.cursor(dictionary=True)

        foreign_keys = {}

        # # Query to get foreign keys
        cursor.execute("""
            SELECT
                TABLE_NAME,
                COLUMN_NAME,
                CONSTRAINT_NAME,
                REFERENCED_TABLE_NAME,
                REFERENCED_COLUMN_NAME
            FROM
                information_schema.KEY_COLUMN_USAGE
            WHERE
                TABLE_SCHEMA = %s
                AND REFERENCED_TABLE_NAME IS NOT NULL;
        """, (config.db_name,))

        for row in cursor.fetchall():
            table_name = row['TABLE_NAME']
            referenced_table_name = row['REFERENCED_TABLE_NAME']
            # column_name = row['COLUMN_NAME']
            # constraint_name = row['CONSTRAINT_NAME']
            # referenced_column_name = row['REFERENCED_COLUMN_NAME' ]

            if table_name not in foreign_keys:
                foreign_keys[table_name] = []

            foreign_keys[table_name].append({
                # 'column_name': column_name,
                'referenced_table_name': referenced_table_name,
            })

        # Close cursor and connection
        cursor.close()
        conn.close()

        return foreign_keys

    def format_table_descriptions(self, table_structure):
        table_descriptions = {}

        for table_name, fields in table_structure.items():
            description = f"{table_name}: "
            field_descriptions = []

            for field in fields:
                field_description = f"{field['Field']}"
                field_descriptions.append(field_description)

            description += ", ".join(field_descriptions)
            table_descriptions[table_name] = description

        return table_descriptions

    def format_foreign_keys(self, foreign_keys):
        formatted_keys = {}
        for table_name, fields in foreign_keys.items():
            related_tables = []
            for field in fields:
                related_table = f"{field['referenced_table_name']}"
                related_tables.append(related_table)
            formatted_keys[table_name] = related_tables
        return formatted_keys

class DbTypeInfo(BaseModel):
    """Database type information."""

    db_type: str = Field(..., description="Database type.")
    is_file_db: bool = Field(False, description="Whether the database is file-based.")

