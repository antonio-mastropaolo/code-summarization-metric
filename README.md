# Evaluating Code Summarization Techniques: A New Metric and an Empirical Characterization

In this study, we explore the suitability of current state-of-the-art code summarization metrics for evaluating the similarity between two pieces of text: the reference and the generated summary. Traditionally, this evaluation is based on comparing the word tokens in the reference with those predicted by a code summarizer. However, we identify two scenarios where this assumption may fall short:

(i) The reference summary, often extracted from code comments in software repositories, may be of low quality due to outdated or less relevant information.

(ii) The generated summary might use different wording compared to the reference but still convey the same meaning, making it suitable to document the code snippet effectively.

To address these limitations, we introduce a new dimension that assesses the extent to which the generated summary aligns with the semantics of the documented code snippet, independently of the reference summary. By empirically evaluating this aspect, we propose a new metric called SIDE (Summary alIgnment to coDe sEmantics) that leverages contrastive learning to better capture the developers' assessment of the automatically generated summaries' quality. 

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
    The models we trained to develop SIDE is publicly available at the following link: <a href ='https://drive.google.com/drive/folders/1eb4C-wjocn_0NEtjIhHIIwDT-Bvr9Yne?usp=sharing'>MODEL</a> 


* ##### Statistical Analysis
    The scripts needed to conduct the analysis are available <a href='https://github.com/antonio-mastropaolo/code-summarization-metric/tree/main/Analysis'>here</a> 
    



    
