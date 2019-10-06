from subprocess import call
import subprocess
import os
import pickle
import key_concept_extraction

# execfile("key_concept_extraction.py")
#
# with open ('final_concepts', 'rb') as fp:
#     temp = pickle.load(fp)

temp = key_concept_extraction.main()

with open('templates/results.html', 'a') as table_file:
	a = '<!DOCTYPE html><html lang="en"><link rel="stylesheet" media="screen" href = "{{ url_for('

	b= "'static'"

	c = ', filename='

	d = "'css/bootstrap.min.css'"

	e = ') }}"><head><meta charset="utf-8"><meta http-equiv="X-UA-Compatible" content="IE=edge"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Content Enrichment</title><style>table, th, td{border:1px solid black;cellpadding: 30px;margin: 5px;}</style><meta name="viewport" content="width=device-width, initial-scale=1"><link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css"><script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script><script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script></head><body><div class="container"><div class="row"><div class="col-md-4"><form action="{{ url_for('

	f = "'give_me_text'"

	with open ('fetched_url_download', 'rb') as fp:
		text = pickle.load(fp) 

	g = ') }}" method="post"><div class="form-group"><label for="comment">Enter Input Text:</label><textarea class="form-control" rows="7" id="comment" name="input_text">'+text+'</textarea></div>  <input type="submit" name="action" class="btn btn-info" style="width:360px" value="Extract Concepts"/><br/><br/></form><form action="{{ url_for('
	h = "'test'"
	i = ') }}" method="post"><div class="form-group"><label for="exampleFormControlSelect1">Extracted Concepts:</label><select class="form-control" name="exampleFormControlSelect1">'

	final_pre = a+b+c+d+e+f+g+h+i
	table_file.write(final_pre)


	for each in temp:
		table_file.write('<option>'+each)
		table_file.write('</option>')

	l ='</select></div></div><div class="col-md-4"><br/><input type="submit" name="action" class="btn btn-info" style="width:360px" value="Extract Prerequisites"/><br/><br/><div class="panel panel-default" style="width:360px;height:300px"><div class="panel-body">Showing your results here!!</div></div></div><div class="col-md-4"><br/><input type="submit" name="action" class="btn btn-info" style="width:360px" value="Extract Definition"/><br/><br/><div class="panel panel-default" style="width:360px;height:150px;margin-bottom:0px;overflow-y:scroll"><div class="panel-body">Showing your results here!!</div></div><br/><input type="submit" name="action" class="btn btn-info" style="width:360px" value="Extract Application"/><br/><br/><div class="panel panel-default" style="width:360px;height:150px;margin-top:0px;overflow-y:scroll"><div class="panel-body">Showing your results here!!</div></div></div></div><hr><div class="row"><div class="col-md-2"></div><div class="col-md-8"><center><h2>Feedback</h2></center><table class="table"><thead><tr><th>#</th><th>Prerequisite Graph</th><th>Defintion Enrichment</th><th>Application Enrichment</th></tr></thead><tbody><tr><th scope="row">1</th><td><input class="form-check-input" type="radio" name="pq" value="ba">Below Average</td><td><input class="form-check-input" type="radio" name="der" value="op1">Enrichment needed and not provided</td><td><input class="form-check-input" type="radio" name="aer" value="op1">Enrichment needed and not provided</td></tr><tr><th scope="row">2</th><td><input class="form-check-input" type="radio" name="pq" value="a">Average</td><td><input class="form-check-input" type="radio" name="der" value="op2">Enrichment needed and also provided</td><td><input class="form-check-input" type="radio" name="aer" value="op2">Enrichment needed and not provided</td></tr><tr><th scope="row">3</th><td><input class="form-check-input" type="radio" name="pq" value="g">Good</td><td><input class="form-check-input" type="radio" name="der" value="op3">Enrichment not needed but provided</td><td><input class="form-check-input" type="radio" name="aer" value="op3">Enrichment needed and not provided</td></tr><tr><th scope="row">4</th><td><input class="form-check-input" type="radio" name="pq" value="e">Excellent</td><td><input class="form-check-input" type="radio" name="der" value="op4">Enrichment not needed and not provided</td><td><input class="form-check-input" type="radio" name="aer" value="op4">Enrichment needed and not provided</td></tr></tbody></table><div class="media-body"><input type="submit" name="action" class="btn btn-info" style="margin-left:16.5em" value="Submit Feedback" disabled="True" /></div></div></div><div class="col-md-2"></div></div> </div></form><script src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.4/jquery.min.js"></script><script type="text/javascript" src="{{ url_for('

	m = "'static'"
	n = ', filename='
	o = "'js/bootstrap.min.js'"
	p = ') }}" ></script></body></html>'

	final_post = l+m+n+o+p
	table_file.write(final_post)

	table_file.close()
