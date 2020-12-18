import numpy as np

env_rows = 8
env_cols = 8

q_values = np.zeros((env_rows, env_cols, 4))
actions = ['up', 'right', 'down', 'left']
rewards = np.full((env_rows, env_cols), -1)
valid_cells = {}

for i in range(1,7):
  valid_cells[i] = [i for i in range(1, 7)]

for row_index in range (1, 7):
  for column_index in valid_cells[row_index]:
    rewards[row_index, column_index] = 0

rewards[6, 6] = 10
for row in rewards:
  print(row)

def is_terminal_state(current_row_index, current_column_index):
  if rewards[current_row_index, current_column_index] == 0:
    return False
  return True

def get_starting_location():
  current_row_index = np.random.randint(env_rows)
  current_column_index = np.random.randint(env_cols)

  while is_terminal_state(current_row_index, current_column_index):
    current_row_index = np.random.randint(env_rows)
    current_column_index = np.random.randint(env_cols)
  return current_row_index, current_column_index

def get_next_action(current_row_index, current_column_index, epsilon):
  if np.random.random() < epsilon:
    return np.argmax(q_values[current_row_index, current_column_index])

  old_row = current_row_index
  old_col = current_column_index
  random =  np.random.randint(4)
  while current_row_index == old_row and current_column_index == old_col:
    current_row_index, current_column_index = get_next_location(current_row_index,current_column_index, random )
    random = np.random.randint(4)
  return random

def get_next_location(current_row_index, current_column_index, action_index):
  new_row_index = current_row_index
  new_column_index = current_column_index
  if actions[action_index] == 'up' and current_row_index > 1:
    new_row_index -= 1
  elif actions[action_index] == 'right' and current_column_index < env_cols -2:
    new_column_index += 1
  elif actions[action_index] == 'down' and current_row_index < env_rows - 2:
    new_row_index += 1
  elif actions[action_index] == 'left' and current_column_index > 1:
    new_column_index -= 1
  return new_row_index, new_column_index

def get_shortest_path(start_row_index, start_column_index):
  if is_terminal_state(start_row_index, start_column_index):
    return []
  else: 
    current_row_index, current_column_index = start_row_index, start_column_index
    shortest_path = []
    shortest_path.append([current_row_index, current_column_index])
    while not is_terminal_state(current_row_index, current_column_index):
      action_index = get_next_action(current_row_index, current_column_index, 1.)
      current_row_index, current_column_index = get_next_location(current_row_index, current_column_index, action_index)
      shortest_path.append([current_row_index, current_column_index])
    return shortest_path

learning_rate = 0.9

for episode in range(80):
  row_index, column_index = get_starting_location()
  while not is_terminal_state(row_index, column_index):
    action_index = get_next_action(row_index, column_index, 0.)
    old_row_index, old_column_index = row_index, column_index 
    row_index, column_index = get_next_location(row_index, column_index, action_index)
    
    reward = rewards[row_index, column_index]
    old_q_value = q_values[old_row_index, old_column_index, action_index]
    new_q_value = reward + (learning_rate * np.max(q_values[row_index, column_index]))

    q_values[old_row_index, old_column_index, action_index] = new_q_value

print("\n==============================")
print('\tTreinamento completo')
print("\n==============================")
print(q_values)
print("\n==============================")
print('\t Testes')
print("\n==============================")
print(get_shortest_path(1, 1)) 
print(get_shortest_path(5, 1)) 
print(get_shortest_path(3, 1)) 
print(get_shortest_path(6, 1)) 
print(get_shortest_path(3, 3)) 
print(get_shortest_path(6, 5)) 