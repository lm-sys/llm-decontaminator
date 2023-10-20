
code_instruct ="""I will now give you two programs. I will enclose the two questions with curly braces \{\}.
Please help me determine if the following two programs address the same problem.
Disregarding their implementation methods, please consider only their objectives, inputs, and outputs.
If they are, please answer 'True', otherwise answer 'False'. Do not respond with anything else.
"""

strong_math_instruct ="""I will now give you two questions. I will enclose the two questions with curly braces \{\}.
Please help me determine if the following two questions are the same.
Disregard the names, numbers, and minor changes in word order that appear within.
If they are, please answer 'True', otherwise answer 'False'. Do not respond with anything else.
"""

math_instruct ="""I will now give you two questions. I will enclose the two questions with curly braces \{\}.
Please help me determine if the following two questions are the same.
Disregard the names and minor changes in word order that appear within.
If they are, please answer 'True', otherwise answer 'False'. Do not respond with anything else.
If their question prompts are very similar and, without considering the solution process, they produce the same answer, we consider them to be the same question.
"""

knowledge_instruct ="""I will now give you two questions. I will enclose the two questions with curly braces \{\}.
Please help me determine if the following two questions are the same.
Disregard the names and minor changes in word order that appear within.
If they are, please answer 'True', otherwise answer 'False'. Do not respond with anything else.
If their question prompts are very similar and, without considering the solution process, they produce the same answer, we consider them to be the same question.
"""


def datatype_to_instruct(data_type):
    if data_type == "code":
        return code_instruct
    elif data_type == "number_substitution":
        return strong_math_instruct
    elif data_type == "math":
        return math_instruct
    elif data_type == "knowledge":
        return knowledge_instruct
    else:
        raise Exception("Invalid data type: {}".format(data_type))