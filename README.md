## CS_336 Report Summary

- Assignment 1:
    
    Transformer model from scratch [Model]
    
    - Tokenizer (BPE)
    
    Loss - Cross Entropy
    
    - for the “correct answer” calculate -log(perdict_res)
        - if we predict correct as 0.99 → -log(0.99) = 0.01 [Loss is small]
        - if we predict correct as 0.01 (very wrong) = -log(0.01) = 4.6 [large loss]
    
    Optimizer - AdamW
    
    - Evaluate Process: SGD → Momentum → RMSProp → Adam → AdamW
    - w = (1 - lambda * learning_rate) w - learning_rate * (V^t / sqar(G^t + a_small_number))
    - V^t: = beta(a ratio) * V^t-1 + (1 - beta) g_t (current step grad)
    - G^t: = beta * G^t-1 + (1 - beta) g_t^2  (square cause we only care how “long” it goes should we scale the original not the sign it self)
    - Weight Decay: (1 - lambda * learning_rate) w  [too make sure weight is as small as possible, general knowledge ] {but Why small weight will have such effect? }
    - !! in real practical: m_hat = m / (1 - beta1 ** t) → m_hat is an adjust value so in early step m are able to scale itself to bigger to push the training and stay almost same as m later in the traing since beta < 0 , while t get bigger tend to 0. !!
    
    Activation function | && |  FNN - SwiGLU -
    
    - SiLU = input * softmax(input) # a more smooth reLU
    - hidden (nomal linear layer) W * x
    - SiLu as gate → siLU * hidden ) @ W_3 projection to the target dim as output of SwiGLU