import os
import pymysql

conn = pymysql.connect(
    host="127.0.0.1",
    user=os.environ.get("db_user"),
    password=os.environ.get("db_password"),
    database="scoredb",
)

cursor = conn.cursor()


def get_database_information()->str:
    """获取当前数据库中的表与表结构的信息"""
    db_info = []

    cursor.execute("show tables")
    table_names = cursor.fetchall()
    for table in table_names:
        cursor.execute(f"show create table {table[0]}")
        result = cursor.fetchall()
        table_info = f"表名: {result[0][0]}\n建表语句: {result[0][1]}"
        db_info.append(table_info)
    return "\n".join(db_info)


def execute_sql(sql_statement:str)->tuple:
    "执行sql语句并返回执行结果"
    # 此处可以优化，应当加入错误处理，且有sql注入的风险
    cursor.execute(sql_statement)
    result = cursor.fetchall()
    return result


if __name__ == "__main__":
    execute_sql("select sno from score where grade >= 90")