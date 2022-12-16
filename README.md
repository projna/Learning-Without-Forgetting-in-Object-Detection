# Learning-Without-Forgetting-in-Object-Detection

<h6>Abstract</h6>
<p name="Abstract">Convolutional Neural Network (CNN) is a powerful tool for object detection
application. While image classifications (e.g. CNN stacks) worked really well
to classify images, object detections took one step further, drawing one or many
bounding boxes of objects of interest. We propose a new approach using Learning
without Forgetting technique on object detection model, which uses only new task
data to train the network while preserving the original capabilities. Our method
performs favorably compared to the Learning Without Forgetting (LwF) model.</p>

<body>
<h6>Introduction</h6>
<p>
Object Detection is the process of identifying objects, scenes or people in an image with a computer
vision framework. The application involves tracking objects, video surveillance, pedestrian detection,
anomaly detection, people Counting, self-driving cars or face detection. Object detection technology
has been a subject to much research and development due to increasing use of images and videos as
data sources and their huge number of applications [1]. The high-level goal of object detection is to
automate the process of recognizing objects from images that are sampled from scenes.

A major challenge for object detection is the limited amount of annotated data currently available for
machine learning models. Object detection datasets [2] often contain ground truth examples for about
a dozen to a hundred classes of objects, while image classification datasets [3] can include upwards
of 100,000 classes. It requires more computational power to train an object detection model. Based
on the problem of detecting object, it is very important to use the previously trained model [4] to fine
tuning for new task without degrading the performance on old tasks (Catastrophic Forgetting) [5][6].
In Learning Without Forgetting(LwF) [7], the authors addressed a problem of adapting a vision
system to a new task while preserving performance on original tasks. In their work [7], authors
demonstrated the efficacy of LwF for image classification problem that can make a considerable
difference in terms of performance on both training and testing sets for image classification. The
authors mentioned in their future work that they would like to do further experiment on semantic
segmentation, detection and problems outside of computer vision. From this concept of their future
work, we applied the LwF on object detection techniques to evaluate the performance of their model.

This paper focuses on two research findings with regards to the object detection problem.<br>
• RQ1: In the object detection problem, how well a model can learn for a new task on a new
dataset if modify old model with new task? <br>
• RQ2: How do the results differ from the authors’ findings?

In this experiment, we used pretrained object detection model called Faster-RCNN [8 ] as a old model
which was trained on COCO dataset1. Then we created a new model by adding a new task on top
of the old model. Then we trained our new model with PennFudan Dataset 2. Finally, we compare
the old and new models using the PennFudan dataset after training and show their result. From
these results, it can be seen that old model does not lose its knowledge from LwF and new model
better learns from previous knowledge than with the old one. In our experiment, old model always
underperforms than new model like LwF.
</p>
<h6>References</h6>
[1] Narayana Darapaneni, Sunilkumar C M, Mukul Paroha, Anwesh Reddy Paduri, Rohit George
Mathew, Namith Maroli, and Rohit Eknath Sawant. Object detection of furniture and home
goods using advanced computer vision. In 2022 Interdisciplinary Research in Technology and
Management (IRTM), pages 1–5, 2022. doi: 10.1109/IRTM54583.2022.9791508.<br>
[2] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr
Dollár, and C. Lawrence Zitnick. Microsoft coco: Common objects in context. In David Fleet,
Tomas Pajdla, Bernt Schiele, and Tinne Tuytelaars, editors, Computer Vision – ECCV 2014,
pages 740–755, Cham, 2014. Springer International Publishing. ISBN 978-3-319-10602-1.<br>
[3] Mark Everingham, S. M. Ali Eslami, Luc Van Gool, Christopher K. I. Williams, John M.
Winn, and Andrew Zisserman. The pascal visual object classes challenge: A retrospective.
International Journal of Computer Vision, 111:98–136, 2014.<br>
[4] Xu Han, Zhengyan Zhang, Ning Ding, Yuxian Gu, Xiao Liu, Yuqi Huo, Jiezhong Qiu, Yuan
Yao, Ao Zhang, Liang Zhang, Wentao Han, Minlie Huang, Qin Jin, Yanyan Lan, Yang Liu,
Zhiyuan Liu, Zhiwu Lu, Xipeng Qiu, Ruihua Song, Jie Tang, Ji-Rong Wen, Jinhui Yuan,
Wayne Xin Zhao, and Jun Zhu. Pre-trained models: Past, present and future. AI Open, 2:
225–250, 2021. ISSN 2666-6510. doi: https://doi.org/10.1016/j.aiopen.2021.08.002. URL
https://www.sciencedirect.com/science/article/pii/S2666651021000231.<br>
[5] Michael McCloskey and Neal J. Cohen. Catastrophic interference in connectionist networks:
The sequential learning problem. Psychology of Learning and Motivation - Advances in Research
and Theory, 24(C):109–165, January 1989. ISSN 0079-7421. doi: 10.1016/S0079-7421(08)
60536-8. Funding Information: The research reported in this chapter was supported by NIH
grant NS21047 to Michael McCloskey, and by a grant from the Sloan Foundation to Neal
Cohen. We thank Sean Purcell and Andrew Olson for assistance in generating the figures, and
Alfonso Caramazza, Walter Harley, Paul Macaruso, Jay McClelland, Andrew Olson, Brenda
Rapp, Roger Rat-cliff, David Rumelhart, and Terry Sejnowski for helpful discussions.<br>
[6] Ian J. Goodfellow, Mehdi Mirza, Da Xiao, Aaron Courville, and Yoshua Bengio. An empirical
investigation of catastrophic forgetting in gradient-based neural networks, 2013. URL https://arxiv.org/abs/1312.6211.<br>
[7] Zhizhong Li and Derek Hoiem. Learning without forgetting, 2016. URL https://arxiv.org/abs/1606.09282.<br>
[8] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object
detection with region proposal networks. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama,
and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 28.
Curran Associates, Inc., 2015. URL https://proceedings.neurips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf. <br>
</body>
