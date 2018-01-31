import csv

def parse_input(input_file):
	data_dict = dict()
	reader = csv.DictReader(open(input_file,'r'))
	#### Reading the metadata into a DICT
	for line in reader:
	    key = line['ID']
	    data_dict[key] = {'file' :  line['FILE']  ,\
	                      'x' : float( line['FACE_X'] ),\
	                      'y' : float( line['FACE_Y'] ),\
	                      'width' : float( line['FACE_WIDTH'] ),\
	                      'height' : float( line['FACE_HEIGHT'] ),\
	                  }
	return data_dict