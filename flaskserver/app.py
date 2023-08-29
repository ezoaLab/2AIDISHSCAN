from flask import Flask, render_template


#create a flask instance
app= Flask(__name__)


# create a route
@app.route('/')

def index():
    #return "<h1>Hello Ezoa!</>"
    return render_template('index.html')

@app.route('/user/<name>')

def user(name):
    #return "<h1> Hello {}</h1>".format(name)
    return render_template("user.html", user_name=name)