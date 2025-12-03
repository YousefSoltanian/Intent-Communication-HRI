---
title: "ASU RISE Lab - Communication through Interaction"
layout: textlay
excerpt: "Intent inference and mutual adaptation in HRI"
sitemap: false
permalink: /interaction_project.html
---

## Communication through Interaction: Mutual Intent Inference and Signaling to Facilitate Human Learning in HRI

### Project Summary 

This project addresses a fundamental challenge in human-robot and multi-agent interactions: enabling effective mutual adaptation and robust intent inference among agents operating under incomplete information. In many real-world scenarios, humans and robots must collaboratively perform complex tasks without explicitly knowing each other's intentions or strategies. Traditional approaches often assume the human agent has complete information or behaves optimally, which can lead to biased predictions, compromised coordination, and potential safety risks. This situation commonly arises in interactive settings such as autonomous vehicles interacting with human drivers, social navigation, or rehabilitation robotics where the assistive robot must teach the human participant how to walk more effectively.

To overcome these limitations, this research introduces a novel, game-theoretic framework designed to explicitly account for the mutual learning dynamics of interacting agents. The central objective is to develop computational methodologies that allow both humans and autonomous systems to simultaneously infer each other's intent in real time and dynamically adapt their actions based on these inferences.

The research approach comprises following key objectives:

1. **Joint Intent Inference**: Develop game-theoretic models that represent mutual inference processes between interacting agents. This involves not only inferring a human agent's immediate intention but also modeling how the human perceives and predicts the robot's intention.

2. **Optimal Control for Active Intent Signaling**: Formulate control strategies for robots to actively communicate their intentions to human counterparts, significantly reducing uncertainty and facilitating smoother, more efficient collaboration.

3. **Modeling Human Learning Dynamics**: Employ experimental data and observational insights to build accurate models of human adaptive behavior in interactive scenarios. Understanding these dynamics is crucial for enabling robots to predict human responses accurately and adjust their assistance strategies appropriately.

4. **Validation with Physical Interaction Tasks**: Apply and test these methodologies through realistic physical interaction experiments, specifically utilizing assistive robotic devices like knee exoskeletons, aiming to demonstrate improvements in collaborative task performance, safety, and comfort.

Through these efforts, the project aims to fundamentally shift how autonomous systems engage with human partners and other robots. Rather than relying on rigid, predefined assumptions, this research promotes dynamic, mutual adaptation frameworks that enhance robots' ability to integrate and collaborate across a range of real-world applications—from assistive rehabilitation devices to collaborative industrial robots.

<figure>
<img src="{{ site.url }}{{ site.baseurl }}/images/respic/overview.png" width="100%">
</figure>

### Project Team
<div class="row">

<div class="col-sm-2 clearfix">
![]({{ site.url }}{{ site.baseurl }}/images/teampic/WL-400450.jpg){: style="width: 111px; float: center; border: 10px"}
Wenlong Zhang, PI
{: style="font-size: 100%; text-align: center;"}
</div>

<div class="col-sm-2 clearfix">
![]({{ site.url }}{{ site.baseurl }}/images/teampic/photo_Yousef.jpg){: style="width: 125px; float: center; border: 10px"}
Seyed Yousef Soltanian, PhD Student
{: style="font-size: 100%; text-align: center;"}
</div>

</div>

<div class="row">

<div class="col-sm-12 clearfix">
<h4>Project Alumni</h4>
{% for member in site.data.alumni_nri %}
{{ member.name }}<br>
{{ member.title }}<br>
{{ member.job }}
{% endfor %}
</div>

</div>

### Publications

- [PACE: A Framework for Learning and Control in Linear Incomplete-Information Differential Games](https://proceedings.mlr.press/v283/soltanian25a.html)  
  Seyed Yousef Soltanian, Wenlong Zhang, *7th Annual Learning for Dynamics & Control Conference (2025)*.

- [Peer-Aware Cost Estimation in Nonlinear General-Sum Dynamic Games for Mutual Learning and Intent Inference](https://arxiv.org/abs/2504.17129)  
  Seyed Yousef Soltanian, Wenlong Zhang, *arXiv preprint (2025)*.

### Acknowledgement and Disclaimer

This material is based upon work supported by the National Science Foundation Grants No. 1944833.  
Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors  
and do not necessarily reflect the views of the National Science Foundation.
