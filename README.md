# PAPER's TITLE

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis consequat nunc nec vehicula egestas. Integer suscipit urna orci, nec luctus libero sollicitudin eu. Donec at vehicula quam. Ut suscipit at neque eget mollis. Proin euismod odio vel quam egestas sodales at sit amet nibh. Maecenas eros dolor, posuere quis laoreet ut, suscipit nec augue. Duis iaculis pharetra tempor. Ut rhoncus orci consequat mattis ultrices. Integer ornare ex in accumsan accumsan. Nunc eu mi non velit feugiat malesuada eget vel quam. Mauris massa ex, fermentum ac lacus id, ultrices vestibulum orci.

#### Pipeline Description

To build SIDE (Summary alIgnment to coDe sEmantics), we relied on Contrastive learning to learn an embedding space where similar sample pairs (i.e., pairs sharing specific features) are clustered together while dissimilar pairs are set apart. In our context, we want to use contrastive learning to teach the model in discriminating textual summaries being suitable for a given code snippet vs summaries which are unsuitable.


#### Contrastive Loss
In our study, we employ the Triplet loss, which have been shown to better encode the positive/negative samples as compared to other contrastive losses . The triplet loss function has been proposed by Schroff et. al  and introduces the concept of "anchor". Given an anchor $x$, a positive ($x^{+}$) and a negative ($x^{-}$) sample is selected, with the triplet loss which during training minimizes the distance between the $x$ and $x^{+}$, while maximizing the distance between $x$ and $x^{-}$:

$$
	E=\max \left(\left\|f_a-f_p\right\|^2-\left\|f_a-f_n\right\|^2+m, 0\right)
$$

In our case, the anchor is the code to document, with a suitable summary representing $x^{+}$ and an unsuitable summary representing $x^{-}$. In the following sections we introduce the dataset used to fine-tune MPNet for the task of interest, explaining how we generate positive and negative samples.


![image info](triplet-example.png)



* ##### Datasets :paperclip:

    The dataset used for training and evaluation are available <a href="https://drive.google.com/drive/folders/1it5-myF8KO8079BfO8IxZ_1IwDLD5EBg?usp=share_link">here</a>
    
    
* ##### Fine-tuned Models :computer:
    The models we trained to develop SIDE are publicly available at the following links:
    -  <a href ='https://drive.google.com/drive/folders/1K3xkEF5UGrLUtBy9IyMLx66svhzfeLKQ?usp=sharing'>Trivial Negatives</a> 
    -  <a href ='https://drive.google.com/drive/folders/1eb4C-wjocn_0NEtjIhHIIwDT-Bvr9Yne?usp=sharing'>Hard Negatives</a> 

* ##### Statistical Analysis
    The scripts needed to conduct the analysis are available <a href=''>here</a> 
    



    
