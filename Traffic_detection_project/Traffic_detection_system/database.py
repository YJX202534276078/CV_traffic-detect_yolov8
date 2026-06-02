# database
import pyodbc
from datetime import datetime

def connect_to_database():
    try:
        conn = pyodbc.connect(
            "Driver={ODBC Driver 17 for SQL Server};"
            "Server=localhost;"
            "Database=Traffic_system_database;"
            "Trusted_Connection=yes;"
        )
        return conn
    except Exception as e:
        print(f"连接数据库失败: {e}")
        return None

def insert_violation_record(conn, class_name, confidence, violation_type):
    if not conn:
        print("数据库连接不可用")
        return

    try:
        cursor = conn.cursor()
        violation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        query = """
            INSERT INTO Traffic_violation_information_table (class_name, confidence, violation_type, violation_time)
            VALUES (?, ?, ?, ?)
        """
        cursor.execute(query, class_name, confidence, violation_type, violation_time)
        conn.commit()
        print("违法记录已插入数据库")
    except Exception as e:
        print(f"插入数据库失败: {e}")

