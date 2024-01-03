# coding:utf-8
import yaml
def get_conf(conf_path):
	with open(conf_path, 'r', encoding='utf-8') as f:
		config = f.read()
	conf = yaml.load(config,Loader=yaml.FullLoader)  # 用load方法转字典
	return conf

if __name__ == "__main__":
	conf = get_conf("../conf/config.yaml")
	print(conf)
	print("".split(","))