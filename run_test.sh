python tl_main.py --path ./dataset/ --aug NO > no_aug.txt
mv ./dataset/test_classified ./dataset/test_classified_no
python tl_main.py --path ./dataset/ --aug RANDOM > rand_aug.txt
mv ./dataset/test_classified ./dataset/test_classified_rand
