from configparser import ConfigParser

# load data
config_file = 'config/config.ini'
config_object = ConfigParser(comment_prefixes=('#', ';'))
config_object.read(config_file)

# objects
data_cfg = config_object['DATA']
model_cfg = config_object['MODEL']
strategy_cfg = config_object['STRATEGY']
client_cfg = config_object['CLIENT']
