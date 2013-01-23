"""
Build the Blaze landing page
"""

import os
import sys
import codecs
import shutil
from jinja2 import Environment, FileSystemLoader
from os.path import abspath, join

#------------------------------------------------------------------------
# Page Variables
#------------------------------------------------------------------------

doc_url  = 'docs/index.html'

src_url  = 'http://github.com/ContinuumIO/blaze-core'
mail_url = 'http://groups.google.com/a/continuum.io/forum/#!forum/blaze-dev'

pages = [
    ("Blaze"         , 'index.html'   , 'index'  , 'home'),
    ("Documentation" , doc_url        , 'docs'   , 'book'),
    ("Examples"      , 'examples.html', 'examples', 'ok'),
    ("Source"        , src_url        , 'source' , 'download') ,
    ("Vision"        , 'vision.html'  , 'vision' , 'star'),
    ("People"        , 'people.html'  , 'people' , 'user'),
    ("Mailing List"  , mail_url       , 'mail'   , 'envelope'),
]

defaultmpl = 'default'
contentdir = 'content'
templatedir = join(os.getcwd(), 'templates')
loader = FileSystemLoader(templatedir)
env = Environment(loader=loader)

#------------------------------------------------------------------------
# Render Templates
#------------------------------------------------------------------------

def build_folder():
    """ Flush and rebuild the _build folder """
    try:
        shutil.rmtree('_build')
    except OSError:
        pass

    os.mkdir('_build')
    shutil.copytree('css', '_build/css')
    shutil.copytree('img', '_build/img')
    return abspath('./_build')

def render_page(bfolder, name, template=None):
    """ Render a tmpl file in terms of a content file """

    if template is None:
        template = defaultmpl

    output = join(bfolder, name + '.html')

    template = env.get_template(template + '.tmpl')
    content  = join(contentdir,name + '.tmpl')
    nav = env.get_template('nav' + '.tmpl')

    input_file = codecs.open(content, "r", encoding="utf-8")
    text = input_file.read()

    ctx = {
        'active_page' : name,
        'nav'         : nav.render(navigation_bar=pages, active_page=name),
        'content'     : text,
        'github'      : src_url,
    }

    with codecs.open(output, 'w+', 'utf-8') as f:
        rendered = template.render(**ctx)
        f.write(rendered)
        f.flush()

if __name__ == '__main__':
    bfolder = build_folder()
    render_page(bfolder, 'index')
    render_page(bfolder, 'vision')
    render_page(bfolder, 'people')
    render_page(bfolder, 'examples')
    print 'Done'
