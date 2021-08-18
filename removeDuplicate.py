import os


files = []
for file in os.listdir():
	p1, p2 = str(file).split('.')
	if p1[-3:] == '(1)':
		os.remove(str(file))
		print('removed', file)



