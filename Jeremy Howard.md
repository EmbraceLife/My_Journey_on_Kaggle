<mark style="background: #FFB8EBA6;">Podcast</mark> 

<mark style="background: #FFB86CA6;">The Simple but Profound Insight Behind Diffusion by W&B</mark>  [video](https://wandb.ai/wandb_fc/gradient-dissent/reports/Jeremy-Howard-The-Simple-but-Profound-Insight-Behind-Diffusion--VmlldzozMjMxODEw?utm_source=youtube&utm_medium=video&utm_campaign=gd-jeremy-howard)



> Jeremy: I have been telling everybody ... we are in the middle of <mark style="background: #FFB8EBA6;">a significant spike (diffusion model specificially) of technological capabilities</mark> right now. If you are not doing that you are missing out on being at the forefront of something that's substantially changing what humans are able to do. [watch](https://youtu.be/HhGOGuJY1Wk?t=141)


> Jeremy: Yeah. It's a <mark style="background: #FFB8EBA6;">simple but profound insight</mark> . It's very difficult for a model to generate something creative, and aesthetic,and correct from nothing...or from nothing but a prompt to a question, or whatever. The profound insight is to say, "Well, given that that's hard, <mark style="background: #ADCCFFA6;">why don't we not ask a modelto do that directly? Why don't we train a model to do something a little bit better than nothing?</mark> And then make a model that — if we run it multiple times — takes a thing that's a little bit better than nothing, and makes that a little bit better still, and a little bit better still." If you run the model multiple times, <mark style="background: #ADCCFFA6;">as long as it's capable of improving the previous output each time, then it's just a case of running it lots of times</mark> . And that's the insight <mark style="background: #FFB8EBA6;">behind diffusion models</mark> . [watch](https://youtu.be/HhGOGuJY1Wk?t=194)


> It's the same basic insight that belongs to this class of models called "<mark style="background: #FFB8EBA6;">boosted models</mark> ". <mark style="background: #ADCCFFA6;">Boosted models are when you train a model to fix a previous model, to find its errors and reduce them</mark> . We use lots of boosted models. Gradient boosting machines in particular are particularly popular, but any model can be turned into a boosted model by training it to fix the previous model's errors. But yeah, <mark style="background: #ADCCFFA6;">we haven't really done that in generative models before</mark> . And we now have <mark style="background: #ADCCFFA6;">a whole infrastructure</mark> for how to do it well. 


> Jeremy: Sure. Broadly speaking, we're looking to create a function that, if we apply it to an input,, it returns a better version of that input. For example, if we try to create a picture that represents "a cute photo of a teddy bear",, then we want a function that takes anything that's not yet "a really great, cute photo of a teddy bear" and makes it something a little bit more like "a cute photo of a teddy, bear" than what it started with. And furthermore, that can take the output of a previous version of running this model, and run it again to create something that's even more like "a cute version of a teddy bear"., It's a little harder than it first sounds, because of this problem of out-of-distribution, inputs. The thing is if the result of running the model once is something that does look a little, bit more like a teddy bear, that output needs to be valid as input to running the model, again. If it's not something the model's been trained to recognize, it's not going to do a good, job. The tricky way that current approaches generally do that, is that they basically do the same, thing that we taught in our 2018-2019 course, which is what we call "crap-ification"., Which is, to take a perfectly good image and make it crappy., In the course, what we did was we added JPEG noise to it, and reduced its resolution, and, scrolled[?] text over the top of it. The approach that's used today is actually much more rigorous, but in some ways less, flexible. It's to sprinkle Gaussian noise all over it. Basically, add or subtract random numbers from every pixel., The key thing is then that one step of inference — making it slightly more like a cute teddy, bear — is basically to "Do your best to create a cute teddy bear, and then sprinkle, a whole bunch of noise back onto the pixels, but a bit less noise than you had before."

That's, by definition, at least going to be pretty close to being in distribution, in

the sense that you train a model that learns to take pictures which have varying amounts

of noise sprinkled over them and to remove that noise. So you could just add a bit less noise, and then you run the model again, and add a bit

of noise back — but a bit less noise — and then run the model again, and add a bit noise back — but a bit less noise — and so forth.

It's really neat. But it's like...a lot of it's done this way because of theoretical convenience, I guess.

It's worked really well because we can use that theoretical convenience to figure out

what good hyperparameters are, and get a lot of the details working pretty well.

But there's totally different ways you can do things. And you can see even in the last week there's been two very significant papers that have

dramatically improved the state of the art. Both of which don't run the same model each time during this boosting phase, during this

diffusion phase. They have different models for different amounts of noise, or there are some which will have

super resolution stages. You're basically creating something small than making it bigger, and you have different models for those.

Basically, what we're starting to see is that gradual move away from the stuff that's theoretically

convenient to stuff that is more flexible, has more fiddly hyperparameters to tune.

But then people are spending more time tuning those hyperparameters, creating a more complex

mixture of experts or ensembles.

I think there's going to be a lot more of that happening. And also, the biggest piece I think will be this whole question of, "Well, how do we use

them with humans in the loop most effectively?" Because the purpose of these is to create stuff, and currently it's almost an accident

that we can ask for a photo of a particular kind of thing, like a cute teddy bear.

The models are trained with what's called "conditioning", where they're conditioned on these captions.

But the captions are known to be wrong, because they come from the alt tags in HTML web pages,

and those alt tags are very rarely accurate descriptions of pictures. So the whole thing...and then the way the conditioning is done has really got nothing

to do with actually trying to create something that will respond to prompts.

The prompts themselves are a bit of an accident, and the conditioning is kind of a bit of an accident. The fact that we can use prompts at all, it's a bit of an accident.

As a result, it's a huge art right now to figure out like, "trending on art station,

8k ultra realistic, portrait of Lukas Biewald looking thoughtful," or whatever.

There's whole books of, "Here's lots of prompts we tried, and here's what the outputs look

like". How do you customize that?

Because, actually, you're trying to create a story book about Lukas Biewald's progress

in creating a new startup, and you want to fit into this particular box here, and you

want a picture of a robot in the background there. How do you get the same style, the same character content, the particular composition?

It's all about this interaction between human and machine. There's so many things which we're just starting to understand how to do.

And so, in the coming years I think it will turn into a powerful tool for computer-assisted

human creativity, rather than what it is now, which is more of a, "Hand something off to

the machine and hope that it's useful."