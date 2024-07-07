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

    def execute_query(self, query: str, params=None):
        # execute a query and return results
        conn = mysql.connector.connect(
            host=self.db_host,
            port=self.db_port,
            user=self.db_user,
            password=self.db_pwd,
            database=self.db_name
        )

        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, params)
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return results

    def get_table_structure(self):
        # retrieve and structure the column information of all tables
        tables = self.execute_query("SHOW TABLES")
        table_structure = {}

        for table in tables:
            table_name = table['Tables_in_' + self.db_name]
            columns = self.execute_query(f"""
                        SELECT COLUMN_NAME, COLUMN_TYPE, COLUMN_COMMENT
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_SCHEMA = '{self.db_name}' AND TABLE_NAME = '{table_name}'
                    """)
            table_structure[table_name] = [
                {'Field': column['COLUMN_NAME'], 'Type': column['COLUMN_TYPE'], 'Comment': column['COLUMN_COMMENT']}
                for column in columns
            ]

        return table_structure

    def get_table_data(self):
        # get data
        tables = self.execute_query("SHOW TABLES")

        table_data = {}

        for table in tables:
            table_name = table['Tables_in_' + self.db_name]
            data = self.execute_query(f"SELECT * FROM {table_name} LIMIT 1")
            table_data[table_name] = data

        return table_data


    def get_create_table_sql(self, tables):
        # return create table sql statements
        create_statements = []

        for table in tables:
            columns = self.execute_query(f"""
                        SELECT COLUMN_NAME, COLUMN_TYPE
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_SCHEMA = '{self.db_name}' AND TABLE_NAME = '{table}'
                    """)
            column_definitions = ", ".join([f"{col['COLUMN_NAME']} {col['COLUMN_TYPE']}" for col in columns])
            create_statement = f"CREATE TABLE {table} ({column_definitions})"
            create_statements.append(create_statement)
        return " ".join(create_statements)

    def get_foreign_keys(self):
        query = """
                    SELECT
                        TABLE_NAME,
                        REFERENCED_TABLE_NAME
                    FROM
                        information_schema.KEY_COLUMN_USAGE
                    WHERE
                        TABLE_SCHEMA = %s
                        AND REFERENCED_TABLE_NAME IS NOT NULL;
                """
        foreign_keys = {}
        rows = self.execute_query(query, (self.db_name,))

        for row in rows:
            table_name = row['TABLE_NAME']
            referenced_table_name = row['REFERENCED_TABLE_NAME']

            if table_name not in foreign_keys:
                foreign_keys[table_name] = []

            foreign_keys[table_name].append({
                'referenced_table_name': referenced_table_name,
            })

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

