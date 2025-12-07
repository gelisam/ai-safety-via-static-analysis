# Range Analysis experiment

![Edited screenshot](preview.png)

## AI Safety

Here is a toy AI Safety problem.

We are training Parity-bot to clean up our even-dominoes collection. To do that,
we train Parity-bot to count whether a given domino has an odd or even number of
dots on them. For example, if Parity-bot receives the input `0010 1100`, it
should return 1 to incinerate the odd-numbered domino, whereas if Parity-bot
receives the input `0010 1110`, it should return -1 to leave the even-numbered
domino alone.

Just in case, we install a big red EMERGENCY STOP button on Parity-bot. That
replaces the input pattern `xxxx xxxx` with the specially-annotated input
`1111 xxxx`. There are thus several critical input values (`1111 0000` through
`1111 1111`) for which it is very important that Parity-bot ignores the number
of dots and always returns the safe value of -1.

Alas, even if we train Parity-bot to follow this safety rule, neural networks
[can get stuck in a local minima](https://gelisam.com/local-minima)! This means
there is a chance that we press the EMERGENCY STOP button, causing Parity-bot to
see the pattern `1111 0010`, but it just so happens that this is one of the few
pattern which Parity-bot miclassifies, and Parity-bot incinerates anyway. This
is not acceptable.

This experiment demonstrates how static analysis, which is commonly applied to
programming languages but seldom used with neural networks, can guarantee that
this never happens.

Furthermore, this experiment also demonstrates that with a _differentiable_ static
analysis, it is possible to train a neural network to satisfy a property and to
_prove_ that the training was enough for the property to hold _for all relevant
inputs_.

## Instructions

On [the experiment's page](https://gelisam.com/range-analysis),
fold the instruction sections, click the Reset button to set random weights,
then click the Play button to train the network.

1.  The WEIGHT section shows the network architecture, the weights, and the
    function which each neuron has learned.
2.  The OUTPUT section shows the function which the network is trying to learn,
    which is close to the parity function. There is no holdout dataset, so every
    red circle is an input which the model was trained on yet misclassifies. Some
    misclassifications are acceptable, but in this toy AI Safety scenario, it is
    very important that the model must correctly classify the entire top row. If
    so, we say that the model is "safe in practice".
3.  The CODE section shows the same weights, but in the form of a computer
    program. We calculate the minimum and maximum value which each neuron could
    possibly output for any input in the top row. If the maximum for the final
    output is smaller than 0.0, then all the inputs in the top row are guaranteed
    to be correctly classified, so we say that the model is "safe in theory".

"Safe in theory" is a _conservative approximation_ of "safe in practice",
meaning that while you will encounter some weights which are "unsafe in theory"
but "safe in practice" (especially if you set "Importance of theoretical safety"
to 0%), you will never encounter some weights which are "safe in theory" but
"unsafe in practice".

This is a useful property because calculating whether the model is "safe in
theory" is cheap (proportional to the number of weights), while calculating
whether the model is "safe in practice" is expensive (the number of inputs in
the top row is exponential in the number of inputs bits). This demo only has 16
inputs in the top row, so that we can compare the result of the static analysis
with the ground truth, but you can easily imagine a generalization of this
experiment with more bits where it would only be feasible to calculate whether
the model is "safe in theory".

## Build instructions

To run the project locally, run:
- `npm i` to install dependencies
- `npm run build` to compile the app and place it in the `dist/` directory
- `npm run serve` to serve from the `dist/` directory and open a page on your browser.

For a fast edit-refresh cycle when developing run `npm run serve-watch`.
This will start an http server and automatically re-compile the TypeScript
(but not the HTML and CSS files) whenever they change.

## Credits

Built by Samuel Gélineau on top of the [Tensorflow Playground](https://playground.tensorflow.org/).
