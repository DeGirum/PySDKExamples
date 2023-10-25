import degirum as dg
import dgtools

cloud_token = dgtools.get_token() # get cloud API access token from env.ini file
cloud_zoo_url = dgtools.get_cloud_zoo_url() # get cloud zoo URL from env.ini file

zoo = dg.connect(dg.CLOUD, cloud_zoo_url, cloud_token) # orca1.1