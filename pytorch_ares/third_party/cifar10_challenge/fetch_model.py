"""Downloads a model, computes its SHA256 hash and unzips it
   at the proper location."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import zipfile
import hashlib

if len(sys.argv) == 1 or sys.argv[1] not in ['natural',
                                             'adv_trained',
                                             'secret']:
  print('Usage: python fetch_model.py [natural, adv_trained]')
  sys.exit(1)

if sys.argv[1] == 'natural':
  url = 'https://www.dropbox.com/s/cgzd5odqoojvxzk/natural.zip?dl=1'
elif sys.argv[1] == 'adv_trained':
  url = 'https://www.dropbox.com/s/g4b6ntrp8zrudbz/adv_trained.zip?dl=1'
else: # fetch secret model
  url = 'https://www.dropbox.com/s/ywc0hg8lr5ba8zd/secret.zip?dl=1'

fname = url.split('/')[-1].split('?')[0]  # get the name of the file

# model download
print('Downloading models')
if sys.version_info >= (3,):
  import urllib.request
  urllib.request.urlretrieve(url, fname)
else:
  import urllib
  urllib.urlretrieve(url, fname)

# computing model hash
sha256 = hashlib.sha256()
with open(fname, 'rb') as f:
  data = f.read()
  sha256.update(data)
print('SHA256 hash: {}'.format(sha256.hexdigest()))

# extracting model
print('Extracting model')
with zipfile.ZipFile(fname, 'r') as model_zip:
  model_zip.extractall()
  print('Extracted model in {}'.format(model_zip.namelist()[0]))
