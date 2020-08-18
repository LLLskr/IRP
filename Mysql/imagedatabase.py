import mysql.connector
from mysql.connector import Error

def write_file(data, filename):
    # Convert binary data to proper format and write it on Hard Disk
    with open(filename, 'wb') as file:
        file.write(data)

def readBLOB(emp_id, before, after):
    print("Reading BLOB data from python_employee table")

    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='image_database',
                                             user='root',
                                             password='LLLlll123')

        cursor = connection.cursor()
        sql_fetch_blob_query = """SELECT * from images where image_id = %s"""

        cursor.execute(sql_fetch_blob_query, (emp_id,))
        record = cursor.fetchall()
        for row in record:
            print("image_id = ", row[0], )
            print("event_id = ", row[1])
            image_before = row[2]
            image_after = row[3]
            print("Storing employee image and bio-data on disk \n")
            write_file(image_before, before)
            write_file(image_after, after)

    except mysql.connector.Error as error:
        print("Failed to read BLOB data from MySQL table {}".format(error))

    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

readBLOB(1, "D:\\Mysql\\image\\Earthquake_Japan_Before_07.JPG",
         "D:\\Mysql\\image\\Earthquake_Japan_After_07.JPG")
readBLOB(2, "D:\\Mysql\\image\\Earthquake_Japan_Before_10.JPG",
         "D:\\Mysql\\image\\Earthquake_Japan_After_10.JPG")