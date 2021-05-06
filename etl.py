import os
import re
import sqlalchemy as database


REGEX_FILENAME = "*.*"
COLUMN_NAMES =["column1", "column2"]
DIRECTORY_NAME = "my/favorite/path"

# specify database configurations 
# TODO this should be done with a .config file
config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'newuser',
    'password': 'newpassword',
    'database': 'test_db'
}

db_user = config.get('user')
db_pwd = config.get('password')
db_host = config.get('host')
db_port = config.get('port')
db_name = config.get('database')

# specify connection string
CONNECTION_STRING = f'mysql+pymysql://{db_user}:{db_pwd}@{db_host}:{db_port}/{db_name}'

def load_files(directory, db):
	for file in glob.glob(os.path.join(dirname, REGEX_FILENAME)):
		load(file, db)


def load(file, db):
	florecence_data = read_florecence_file(file)
	table=os.path.splitext(os.path.basename(file))[0]

	sql = 'drop table if exists "{}"'.format(table)
        db.execute(sql)

        sql = 'create table "{table}" ( {cols} )'.format(
            table=table,
            cols=','.join('"{}"'.format(col) for col in COLUMN_NAMES))))
        db.execute(sql)

	# Actually populating the table with the data
        sql = 'insert into "{table}" values ( {vals} )'.format(
            table=table,
            vals=','.join('?' for col in COLUM_NAMES)
	db.executemany(sql, (list(map(row.get, cols)) for row in fluorencence_data))


def read_fluorence_file(file):
	with open(file) as f:
		data = LOAD_THE_FILE_INTO_PYTHON_OBJECT() #TODO NEEDS WORK
	return data


# TODO
def LOAD_THE_FILE_INTO_PYTHON_OBJECT():
	pass


if __name__ == '__main__':
	engine = database.create_engine(CONNECTION_STRING)
	conn = engine.connect()
	load_files(DIRECTORY_NAME, conn)
