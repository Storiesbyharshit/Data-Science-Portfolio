# Machine Learning

Machine Learning is the field of study that gives computers the capability to learn without being explicitly programmed. ML is one of the most exciting technologies that one would have ever come across. As it is evident from the name, it gives the computer that makes it more similar to humans: The ability to learn.


### Classification of Machine Learning

Machine learning implementations are classified into three major categories, depending on the nature of the learning “signal” or “response” available to a learning system which are as follows:-

* **Supervised learning** : When an algorithm learns from example data and associated target responses that can consist of numeric values or string labels, such as classes or tags, in order to later predict the correct response when posed with new examples comes under the category of Supervised learning. This approach is indeed similar to human learning under the supervision of a teacher. The teacher provides good examples for the student to memorize, and the student then derives general rules from these specific examples.
* **Unsupervised learning** :Whereas when an algorithm learns from plain examples without any associated response, leaving to the algorithm to determine the data patterns on its own. This type of algorithm tends to restructure the data into something else, such as new features that may represent a class or a new series of un-correlated values. They are quite useful in providing humans with insights into the meaning of data and new useful inputs to supervised machine learning algorithms.
As a kind of learning, it resembles the methods humans use to figure out that certain objects or events are from the same class, such as by observing the degree of similarity between objects. Some recommendation systems that you find on the web in the form of marketing automation are based on this type of learning.
* **Reinforcement learning** : Reinforcement learning (RL) is an area of machine learning concerned with how software agents ought to take actions in an environment in order to maximize the notion of cumulative reward. Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning. Reinforcement learning is the training of machine learning models to make a sequence of decisions. The agent learns to achieve a goal in an uncertain, potentially complex environment. In reinforcement learning, an artificial intelligence faces a game-like situation. The computer employs trial and error to come up with a solution to the problem. To get the machine to do what the programmer wants, the artificial intelligence gets either rewards or penalties for the actions it performs. Its goal is to maximize the total reward.


Another categorization of machine learning tasks arises when one considers the desired output of a machine-learned system:

* **Classification** : When inputs are divided into two or more classes, and the learner must produce a model that assigns unseen inputs to one or more (multi-label classification) of these classes. This is typically tackled in a supervised way. Spam filtering is an example of classification, where the inputs are email (or other) messages and the classes are “spam” and “not spam”.
* **Regression** : Which is also a supervised problem, A case when the outputs are continuous rather than discrete.
* **Clustering** : When a set of inputs is to be divided into groups. Unlike in classification, the groups are not known beforehand, making this typically an unsupervised task.






##### Basic Difference in ML and Traditional Programming?

> **Traditional Programming : We feed in DATA (Input) + PROGRAM (logic), run it on machine and get output.**

> **Machine Learning : We feed in DATA(Input) + Output, run it on machine during training and the machine creates its own program(logic), which can be evaluated while testing.**
 


A computer is said to be learning from Experiences with respect to some class of Tasks, if its performance in a given Task improves with the Experience.

__A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E__


How ML works?

* Gathering past data in any form suitable for processing.The better the quality of data, the more suitable it will be for modeling
* Data Processing – Sometimes, the data collected is in the raw form and it needs to be pre-processed.
Example: Some tuples may have missing values for certain attributes, an, in this case, it has to be filled with suitable values in order to perform machine learning or any form of data mining.
* Missing values for numerical attributes such as the price of the house may be replaced with the mean value of the attribute whereas missing values for categorical attributes may be replaced with the attribute with the highest mode. This invariably depends on the types of filters we use. If data is in the form of text or images then converting it to numerical form will be required, be it a list or array or matrix. Simply, Data is to be made relevant and consistent. It is to be converted into a format understandable by the machine
* Divide the input data into training,cross-validation and test sets. The ratio between the respective sets must be 6:2:2
* Building models with suitable algorithms and techniques on the training set.
* Testing our conceptualized model with data which was not fed to the model at the time of training and evaluating its performance using metrics such as F1 score, precision and recall.


<p align="center">
  <img alt="For-the-Badge-Python" src="https://miro.medium.com/max/1986/1*iWkJS33mwmOprcKiQtaLaA.jpeg">
  
</p>

#### Workflow 
We can define the machine learning workflow in following stages.
* Gathering data
* Data pre-processing
* Researching the model that will be best for the type of data
* Training and testing the model
* Evaluation

<p align="center">
  <img alt="For-the-Badge-Python" src="https://static.packt-cdn.com/products/9781783980284/graphics/230f1938-fd69-424e-b94e-225038e27c37.png">
  
</p>



###### Difference between Machine learning and Artificial Intelligence
> Artificial Intelligence : The word Artificial Intelligence comprises of two words “Artificial” and “Intelligence”. Artificial refers to something which is made by human or non natural thing and Intelligence means ability to understand or think. There is a misconception that Artificial Intelligence is a system, but it is not a system .AI is implemented in the system. There can be so many definition of AI, one definition can be “It is the study of how to train the computers so that computers can do things which at present human can do better.”Therefore It is a intelligence where we want to add all the capabilities to machine that human contain.

> Machine Learning : Machine Learning is the learning in which machine can learn by its own without being explicitly programmed. It is an application of AI that provide system the ability to automatically learn and improve from experience. Here we can generate a program by integrating input and output of that program. One of the simple definition of the Machine Learning is “Machine Learning is said to learn from experience E w.r.t some class of task T and a performance measure P if learners performance at the task in the class as measured by P improves with experiences.”



ARTIFICIAL INTELLIGENCE |	MACHINE LEARNING
----------- | -------------
AI stands for Artificial intelligence, where intelligence is defined acquisition of knowledge intelligence is defined as a ability to acquire and apply knowledge.	 | ML stands for Machine Learning which is defined as the acquisition of knowledge or skill
The aim is to increase chance of success and not accuracy.	 | The aim is to increase accuracy, but it does not care about success
It work as a computer program that does smart work | 	It is a simple concept machine takes data and learn from data.
The goal is to simulate natural intelligence to solve complex problem | 	The goal is to learn from data on certain task to maximize the performance of machine on this task.
AI is decision making.	 | ML allows system to learn new things from data.
It leads to develop a system to mimic human to respond behave in a circumstances. | 	It involves in creating self learning algorithms.
AI will go for finding the optimal solution. | 	ML will go for only solution for that whether it is optimal or not.
AI leads to intelligence or wisdom. | 	ML leads to knowledge.



<p align="center">
  <img alt="For-the-Badge-Python" src="https://miro.medium.com/max/650/1*-XKVI5SAEpffNR7BusdvNQ.png">
  
</p>

<p align="center">
  <img alt="For-the-Badge-Python" src="http://ForTheBadge.com/images/badges/made-with-python.svg">
  
</p>
