# coding:utf-8
import yaml
def get_conf(conf_path):
	with open(conf_path, 'r', encoding='utf-8') as f:
		config = f.read()
	conf = yaml.load(config,Loader=yaml.FullLoader)  # 用load方法转字典
	return conf

def get_feature_conf(yaml_path):
    """
    type  1:sparse,  2:dense,  3:var
    cate  1:user, 2:item
    """
    conf = get_conf(yaml_path)

    user_sparse_f, item_sparse_f, user_var_f, item_var_f, user_dense_f, item_dense_f = {}, {}, {}, {}, {}, {}
    default_size = 1000
    for feature in conf.items():
        tmp = feature[1]
        if tmp['type'] == 1:
            if tmp['cate'] == 1:
                user_sparse_f.update(
                    {feature[0]: {'size': tmp.get('size', default_size), 'shared': tmp.get('shared', '')}})
            elif tmp['cate'] == 2:
                item_sparse_f.update(
                    {feature[0]: {'size': tmp.get('size', default_size), 'shared': tmp.get('shared', '')}})
        elif tmp['type'] == 2:
            if tmp['cate'] == 1:
                user_dense_f.update(
                    {feature[0]: {'size': tmp.get('size', default_size), 'shared': tmp.get('shared', '')}})
            elif tmp['cate'] == 2:
                item_dense_f.update(
                    {feature[0]: {'size': tmp.get('size', default_size), 'shared': tmp.get('shared', '')}})
        elif tmp['type'] == 3:
            if tmp['cate'] == 1:
                user_var_f.update(
                    {feature[0]: {'size': tmp.get('size', default_size), 'shared': tmp.get('shared', '')}})
            elif tmp['cate'] == 2:
                item_var_f.update(
                    {feature[0]: {'size': tmp.get('size', default_size), 'shared': tmp.get('shared', '')}})

    return user_sparse_f, item_sparse_f, user_var_f, item_var_f, user_dense_f, item_dense_f



if __name__ == "__main__":
	conf = get_conf("../conf/config.yaml")
	print(conf)