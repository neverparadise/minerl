def make_6action(env, action_index):
    action = env.action_space.noop()
    if action_index == 0:
        action['forward'] = 1
    elif action_index == 1:
        action['jump'] = 1
    elif action_index == 2:
        action['camera'] = [0, -5]
    elif action_index == 3:
        action['camera'] = [0, 5]
    elif action_index == 4:
        action['camera'] = [-5, 0]
    elif action_index == 5:
        action['camera'] = [5, 0]

    return action


def make_9action(env, action_index):
    # Action들을 정의
    action = env.action_space.noop()
    if (action_index == 0):
        action['camera'] = [0, -5]
        action['attack'] = 1
    elif (action_index == 1):
        action['camera'] = [0, 5]
        action['attack'] = 1
    elif (action_index == 2):
        action['camera'] = [-5, 0]
        action['attack'] = 1
    elif (action_index == 3):
        action['camera'] = [5, 0]
        action['attack'] = 1
    elif (action_index == 4):
        action['forward'] = 1
        action['jump'] = 1
    elif (action_index == 5):
        action['forward'] = 1
        action['attack'] = 1
    elif (action_index == 6):
        action['back'] = 1
        action['attack'] = 1
    elif (action_index == 7):
        action['left'] = 1
        action['attack'] = 1
    elif (action_index == 8):
        action['right'] = 1
        action['attack'] = 1

    return action

def make_19action(env, action_index):
    # Action들을 정의
    action = env.action_space.noop()
    if (action_index == 0):
        action['camera'] = [0, -5]
        action['attack'] = 0
    elif (action_index == 1):
        action['camera'] = [0, -5]
        action['attack'] = 1
    elif (action_index == 2):
        action['camera'] = [0, 5]
        action['attack'] = 0
    elif (action_index == 3):
        action['camera'] = [0, 5]
        action['attack'] = 1
    elif (action_index == 4):
        action['camera'] = [-5, 0]
        action['attack'] = 0
    elif (action_index == 5):
        action['camera'] = [-5, 0]
        action['attack'] = 1
    elif (action_index == 6):
        action['camera'] = [5, 0]
        action['attack'] = 0
    elif (action_index == 7):
        action['camera'] = [5, 0]
        action['attack'] = 1

    elif (action_index == 8):
        action['forward'] = 0
        action['jump'] = 1
    elif (action_index == 9):
        action['forward'] = 1
        action['jump'] = 1
    elif (action_index == 10):
        action['forward'] = 1
        action['attack'] = 0
    elif (action_index == 11):
        action['forward'] = 1
        action['attack'] = 1
    elif (action_index == 12):
        action['back'] = 1
        action['attack'] = 0
    elif (action_index == 13):
        action['back'] = 1
        action['attack'] = 1
    elif (action_index == 14):
        action['left'] = 1
        action['attack'] = 0
    elif (action_index == 15):
        action['left'] = 1
        action['attack'] = 1
    elif (action_index == 16):
        action['right'] = 1
        action['attack'] = 0
    elif (action_index == 17):
        action['right'] = 1
        action['attack'] = 1
    else:
        action['attack'] = 1

    return action
