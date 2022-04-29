from helpers.utils import convert_txt_to_paths
from helpers.disjoint_set import disjoint_set, check_key_in_set
from pprint import pprint


if __name__ == '__main__':
    train_filename = "data/train_triplets.txt"
    paths = convert_txt_to_paths(train_filename)
    print(len(paths))
    path2root, root2set = disjoint_set(paths)
    check_key_in_set(root2set)
    pprint(list(path2root.items())[:5])
    print("-" * 100)
    # pprint(list(root2set.items())[:5])
    print(len(path2root))
    for key, val in root2set.items():
        print(f"{key}: {len(val)}")
