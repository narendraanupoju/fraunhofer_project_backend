import cv2
import numpy as np
import sqlite3

def convertToBinaryData(filename):
    #Convert digital data to binary format
    with open(filename, 'rb') as file:
        blobData = file.read()
    return blobData

def insertBLOB(name, image):
    try:
        sqliteConnection = sqlite3.connect('Database.db')
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")
        name = 'img'
        cursor.execute("""CREATE TABLE IF NOT EXISTS new_table(image_id INTEGER PRIMARY KEY, name TEXT, image BLOB)""")
        sqlite_insert_blob_query = """ INSERT INTO 'new_table'
                                  ('name', 'image') VALUES (?, ?)"""

        #image = convertToBinaryData(image)
        # Convert data into tuple format
        data_tuple = (name, image)
        cursor.execute(sqlite_insert_blob_query, data_tuple)
        sqliteConnection.commit()
        print("Image inserted successfully as a BLOB into a table")
        cursor.close()

    except sqlite3.Error as error:
        print("Failed to insert blob data into sqlite table", error)

    finally:
        if (sqliteConnection):
            sqliteConnection.close()
            print("the sqlite connection is closed")

#path1 = r'D:\Project\Window Images\xyz.jpg'
#insertBLOB('window2', path1)