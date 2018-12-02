# keras-oneshot
![oneshot task](images/task_25.png)
[koch et al, Siamese Networks for one-shot learning,](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)  reimplimented in keras. 
Trains on the [Omniglot dataset]( https://github.com/brendenlake/omniglot).

Also check out the code's original author's [blog post](https://sorenbouma.github.io/blog/oneshot) about this paper and one shot learning in general!



## Installation Instructions


To run, you'll first have to clone this repo and install the dependencies

```bash
git clone https://github.com/gyani91/Machine-Learning
cd Machine-Learning/One-Shot-Learning
sudo pip install -r requirements.txt

```


Then you'll need to download the omniglot dataset and preprocess/pickle it with the load_data.py script.
```bash
git clone https://github.com/brendenlake/omniglot
python load_data.py --path <PATH TO THIS FOLDER>
```

Now run the jupyter notebook!

