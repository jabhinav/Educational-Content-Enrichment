from subprocess import call
import subprocess
import os
import pickle
import nltk

# execfile("app_extract.py")
exec(compile(open('app_extract.py', 'rb').read(), 'app_extract.py', 'exec'))

with open ('app_in_text', 'rb') as fp:
	app_in_text = pickle.load(fp)

with open ('app_in_wiki', 'rb') as fp:
	app_in_wiki = pickle.load(fp)

with open ('applications', 'rb') as fp:
	applications = pickle.load(fp)

with open ('fetched_url_download', 'rb') as fp:
	text = pickle.load(fp)

with open ('def', 'rb') as fp:
	defin = pickle.load(fp)

with open ('def_check', 'rb') as fp:
	check = pickle.load(fp)

if check == "False": 
	def_text = defin
elif check == "True":
	def_text = ""

final_highlighted_text = ''
app_text = ''
if app_in_text == True:
	sent_text = nltk.sent_tokenize(text)
	for ix in sent_text:
		for iy in applications:
			if ix == iy:
				final_highlighted_text += '<span style="background-color: #ADD8E6">'+ix+'</span>'
			else:
				final_highlighted_text += '<span style="background-color: #FFFFFF">'+ix+'</span>'

elif app_in_text == False and app_in_wiki == True:
	for ix in applications:
		app_text += ix

	final_highlighted_text = text
else:
	print("No definition found")

with open ('final_concepts', 'rb') as fp:
    temp = pickle.load(fp) 

with open('templates/results2.html', 'a') as table_file:
	a = '<!DOCTYPE html><html lang="en"><link rel="stylesheet" media="screen" href = "{{ url_for('

	b= "'static'"

	c = ', filename='

	d = "'css/bootstrap.min.css'"

	e = ') }}"><head><meta charset="utf-8"><meta http-equiv="X-UA-Compatible" content="IE=edge"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Content Enrichment</title><style>table, th, td{border:1px solid black;cellpadding: 30px;margin: 5px;}</style><meta name="viewport" content="width=device-width, initial-scale=1"><link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css"><script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script><script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script></head><body><div class="container"><div class="row"><div class="col-md-4"><form action="{{ url_for('

	f = "'give_me_text'"

	# k = ') }}" method="post"><div class="form-group"><label for="comment">Enter Input Text:</label><textarea class="form-control" rows="7" id="comment" name="input_text">'+final_highlighted_text+'</textarea></div><input type="submit" name="action" class="btn btn-info" style="width:360px" value="Extract Concepts"/><br/><br/></form><form action="{{ url_for('
	g = ') }}" method="post"><div style="overflow-y:scroll;height:200px;" class="form-group"><label for="comment">Enter Input Text:</label><p id="comment" name="input_text">'+final_highlighted_text+'</p></div><input type="submit" name="action" class="btn btn-info" style="width:360px" value="Extract Concepts"/><br/><br/></form><form action="{{ url_for('

	h = "'test'"
	i = ') }}" method="post"><div class="form-group"><label for="exampleFormControlSelect1">Extracted Concepts:</label><select class="form-control" name="exampleFormControlSelect1">'

	final_pre = a+b+c+d+e+f+g+h+i
	table_file.write(final_pre)


	for each in temp[2:]:
		table_file.write('<option>'+each)
		table_file.write('</option>')

	# app_text = 'Applications range from tasks such as industrial machine vision systems which, say, inspect bottles speeding by on a production line, to research into artificial intelligence and computers or robots that can comprehend the world around them. Computer vision covers the core technology of automated image analysis which is used in many fields.'

	# <div class="panel panel-default" style="width:360px;height:300px"><div class="panel-body">Showing your results here!!</div></div>

	l ='</select></div></div><div class="col-md-4"><br/><input type="submit" name="action" class="btn btn-info" style="width:360px" value="Extract Prerequisites"/><br/><br/>    <img style="width:360px;height:300px" src="{{ url_for('
	m = "'static'"
	n = ', filename='
	o = "'model.jpeg'"
	p = ') }}">    </div><div class="col-md-4"><br/><input type="submit" name="action" class="btn btn-info" style="width:360px" value="Extract Definition"/><br/><br/><div class="panel panel-default" style="width:360px;height:150px;margin-bottom:0px;overflow-y:scroll"><div class="panel-body">'+def_text+'</div></div><br/><input type="submit" name="action" class="btn btn-info" style="width:360px" value="Extract Application"/><br/><br/><div class="panel panel-default" style="width:360px;height:150px;margin-top:0px;overflow-y:scroll"><div class="panel-body">'+app_text+'</div></div></div></div><hr><div class="row"><div class="col-md-2"></div><div class="col-md-8"><center><h2>Feedback</h2></center><table class="table"><thead><tr><th>#</th><th>Prerequisite Graph</th><th>Defintion Enrichment</th><th>Application Enrichment</th></tr></thead><tbody><tr><th scope="row">1</th><td><input class="form-check-input" type="radio" name="pq" value="ba">Below Average</td><td><input class="form-check-input" type="radio" name="der" value="op1">Enrichment needed and not provided</td><td><input class="form-check-input" type="radio" name="aer" value="op1">Enrichment needed and not provided</td></tr><tr><th scope="row">2</th><td><input class="form-check-input" type="radio" name="pq" value="a">Average</td><td><input class="form-check-input" type="radio" name="der" value="op2">Enrichment needed and also provided</td><td><input class="form-check-input" type="radio" name="aer" value="op2">Enrichment needed and not provided</td></tr><tr><th scope="row">3</th><td><input class="form-check-input" type="radio" name="pq" value="g">Good</td><td><input class="form-check-input" type="radio" name="der" value="op3">Enrichment not needed but provided</td><td><input class="form-check-input" type="radio" name="aer" value="op3">Enrichment needed and not provided</td></tr><tr><th scope="row">4</th><td><input class="form-check-input" type="radio" name="pq" value="e">Excellent</td><td><input class="form-check-input" type="radio" name="der" value="op4">Enrichment not needed and not provided</td><td><input class="form-check-input" type="radio" name="aer" value="op4">Enrichment needed and not provided</td></tr></tbody></table><div class="media-body"><input type="submit" name="action" class="btn btn-info" style="margin-left:16.5em" value="Submit Feedback" disabled="True" /></div></div></div><div class="col-md-2"></div></div> </div></form><script src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.4/jquery.min.js"></script><script type="text/javascript" src="{{ url_for('

	q = "'static'"
	r = ', filename='
	s = "'js/bootstrap.min.js'"
	t = ') }}" ></script></body></html>'

	final_post = l+m+n+o+p+q+r+s+t
	table_file.write(final_post)

	table_file.close()
