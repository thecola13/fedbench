I want to create a series of Python classes that allow me to play a bit with federated learning, specifically in a setting where we encrypt the models. The general idea is the following:
- A series of parties, each with its own data, wants to compute a model trained on all data, but without sharing their own data to others.
- Therefore, they act as follows: each of them computes their own model, which is then sent to a central "orchestrator" which aggregates them and returns them. 
- However, the model the orchestrator receives might be encrypted (specifically, through FHE) to guarantee privacy of each parties' model. 

Want I want you to do is to generate the required classes for such a playground, such that they allow flexibility in how they act: we might have different types of regression model, different types of encrytions, different types of aggregations. I also would like to able to extract anallytics from the various passages, such as accuracy of each party model, accuracy of the aggregated model after each exchange etc. 

The final code should be as simple as possible to use, and should be able to be extended in the future.

The classes I think we need are the following:
- Environment: it initializes the parties, the orchestrator, and the data. Specifically, it should be able to load the data, and initialize the parties with their respective data. 
- Party: it should be able to compute different types of regressors, and in case do everything is needed to encrypt the model and send it to the orchestrator.
- Orchestrator: it should be able to receive the encrypted models from the parties, and aggregate them to compute the final model. It then returns the final model to the parties.
- Encryption/decryption scheme: as multiple exchanges might happen between parties and orchestrator

Produce the required classes, any other class that you deem necessary and a script showcasing how to connect them together in a simple environment: linear regression, no encryption, models are aggregated through averaging of the parameters. Also, I would like you to document the code thoroughly, and produce a README that clearly explains the functioning of the code. 