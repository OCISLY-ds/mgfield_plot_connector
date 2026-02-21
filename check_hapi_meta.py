from hapiclient import hapi
import json

server = 'https://imag-data.bgs.ac.uk/GIN_V1/hapi'
dataset = 'wng/best-avail/PT1M/xyzf'
parameters = 'Field_Vector'

meta = hapi(server, dataset, parameters, logging=True)
print(json.dumps(meta, indent=2))
