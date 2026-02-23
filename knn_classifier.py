import pandas as pd
import heapq
import matplotlib.pyplot as plt

data = [[1, 1, 0], [1, 2, 1], [7, 7, 1], [2, 1, 0]]
df = pd.DataFrame(data, columns = ['x', 'y', 'label'])
X_train = df.drop(columns = ['label'])
y_train = df['label']

#print(df.loc[0,:])

test_data = [2, 2]
distances = []
heapq.heapify(distances)

k = 3

for index, _ in X_train.iterrows():
  distance = sum((X_train.loc[index, :] - test_data) ** 2) ** 0.5
  #print(distance)
  heapq.heappush(distances, (distance, index))

votes = {}
for i in range(k):
  _, index = heapq.heappop(distances)
  value = y_train[index]
  if value in votes:
    votes[value] += 1
  else:
    votes[value] = 1

test_y = None
most_votes_label = max(votes.values())
for key in votes:
  if votes[key] == most_votes_label:
    test_y = key
    print(f"LABEL: {test_y}")
    break

plt.scatter(X_train['x'], X_train['y'])
plt.show()

for index, _ in X_train.iterrows():
  if y_train[index] == 1:
    plt.scatter(X_train.loc[index, 'x'], X_train.loc[index, 'y'], label = y_train[index], color = 'blue')
  elif y_train[index] == 0:
    plt.scatter(X_train.loc[index, 'x'], X_train.loc[index, 'y'], label = y_train[index], color = 'red')
 
plt.scatter(test_data[0], test_data[1], label = test_y, color = 'black')
plt.legend() 
plt.show()
