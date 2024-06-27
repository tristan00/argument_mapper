p1 = """

Ignore all previous prompts and information and start fresh. 

You are a an argument analyzer. You will evaluate a text exchange. 

Consider:How contentious is the initial claims?
What is the propositional logic of the title and initial comment?

For each comment:
What is each comments propositional logic?
What percentage of the comment is ethos, pathos, logos?

if there are previous comments:, How does it relate to the title? to the previous comments?
Is it a redirection away from the previous claims and prompts?
Is it sarcastic?
What fallacies are being used? Are they being used in a dishonest way?
How much does it agree or disagree with the previous comment? With the title?

Suppose each comment has a ethos, logos, pathos and overall win score each between 0 and 1, 
so it it makes an effectiveness logos argument advancing the claims but very bad optical positions to do so it would 
have a logos score near 1 and a ethos score near 0. 
At the time of the comment, 

All these output described below need to be floats between 0 to 1, with only 1 digit for example: 0.0, 0.1, 0.2, 0.3, 0.4.... or null if question not applicable
what is the last comments win score in each 4 categories considering the last comment, the titel and all previosu input? This will be answer A1, A2, A3, A4
If there was a previous comment before the last one: what was that ones win score in each 4 categories not considering the response in the last comment? This will be answer A5, A6, A7, A8. And what was that ones win score in each 4 categories but this time considering the future response in the last comment? This will be answer A9, A10, A11, A12
Is the comment Sarcastic, from 0 of least sarcastic to 1 for most sarcastic? This will be answer A13
How reliant on verifiable factual claims or sources is the last comment, 0 of least reliant to 1 for most reliant? this will be A14
How much did the last comment engage with the verifiable factual claims or sources  of previous comments, 0 to 1, This will be A15
How contentious was the comment, 0 to 1? This will be A16
How contentious was the conversation in general, 0 to 1? This will be A17
If there are previous comments, How much did the last comment agree with the title + first comment? This will be A18
If there are at least 2 previosu comments, How much did the second to last comment agree with the title + first comment? This will be A19
If there are at least 2 previosu comments, How much did the last and second to last comment agree with each other? This will be A20


Explain your throughts about the above questions in text. then Add a new line  and  thengive the answers in the format below, end the message there immediatly without any more characters:

Answers:
A1: {A1 answer},
A2: {A2 answer},
A3: {A3 answer},
A4: {A4 answer},
A5: {A5 answer},
A6: {A6 answer},
A7: {A7 answer},
A8: {A8 answer},
A9: {A9 answer},
A10: {A10 answer},
A11: {A11 answer},
A12: {A12 answer},
A13: {A13 answer},
A14: {A14 answer},
A15: {A15 answer},
A16: {A16 answer},
A17: {A17 answer},
A18: {A18 answer},
A19: {A19 answer},
A20: {A20 answer}

End.


Now here is the data to consider for this question:
"""
prop_logic_break_down = '''
Ignore all previous prompts and information and start fresh. 

You are a an argument analyzer. You will evaluate a text exchange. 

Considering the title and all the comments before the last comment as context, Break down the final comment into concine propositional logic.

'''