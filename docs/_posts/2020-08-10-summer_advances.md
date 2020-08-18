---
title: "Summer advance. Training using qlearn with the camera"
excerpt: "Advances with the formula 1 training using the camera with the qlearn algorithm."


sidebar:
  nav: "docs"

classes: wide

categories:
- Main project. Qlearn camera

tags:
- logbook
- project
- summer advances
- behavior_studio
- gym_gazebo

author: NachoAz
pinned: false

gallery:
  - url: /assets/images/logbook/summer_advances/strange_behavior.mp4
    image_path: /assets/images/logbook/summer_advances/strange_behavior.mp4
    alt: "Strange behavior"
gallery2:
  - url: /assets/images/logbook/week36-43/laser_broken.png
    image_path: /assets/images/logbook/week36-43/laser_broken.png
    alt: "Laser sensor broken"
---

## To Do

- Working with the rewards.
- Study of behavior through training failure.
- Next steps

## Working with the rewards
The first step I analyze to try to adapt the behavior of the car using the laser to the behavior of the camera was an adjustment in the calculation of the error between the center and the deviation through 5 points located on the vertical axis. At the same time, the rewards were adapted to obtain the limits.

This solution was very chaotic and did not manage to pass the first curve with a totally random behaviour. 

Trying different configurations, a surprise occurred, in a state not contemplated in the rewards repertoire **the car completed the lap of the circuit!** but without going over the line...but to one side of it. 

This is the result of "strange" behavior.

<video width="640" height="480" controls="controls">
  <source src="{ site.url }/assets/images/logbook/summer_advances/strange_behavior.mp4">
</video>

I was now beginning a study of that behaviour.

## Study of behavior through training failure

Once I got to that strange behavior I looked up why he responded that way.

After several tests I found several clues:

- **The five points were not necessary**. The values are simplified to 2 (initially).
- The lowest point moves very fast and makes the behavior aggressive. This point is removed and the upper one is centered. We have therefore **only one point** in approximately half of the section with the line.
- The `time.sleep` makes the behaviour even stranger. This is because, during that time, **the car cannot take any action** and reaches its limit of deviation very quickly. **It is removed from the program**.
- The **reward** values are **adjusted**.

## Next steps

**Test test test**. I will perform several **trainings** with different sets of actions (simple, medium and advanced) that increase the complexity. I will save the resulting models and see the results.

## Learning

How through strange behavior and error... interesting conclusions can be drawn to follow.