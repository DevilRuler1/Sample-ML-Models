import pandas as pd
import matplotlib.pyplot as plt


def import_data():
    df = pd.read_excel(r'Damages rate.xlsx')
    df = df.replace(['Y', 'F'], 1)
    df = df.replace(['N', 'B'], 0)
    df = df.replace('S', 2)
    df['Total Bid Amount'] = df['Total Bid Amount'].apply(lambda x: x / 1000000)
    df['Net Change Order Amount'] = df['Net Change Order Amount'].apply(lambda x: x / 1000000)
    df = df.astype('float32')
    return df


def split_data(data):
    X = data.values[:, :8]
    Y = data.values[:, 8]
    return X, Y


dataFrame = import_data()
X, Y = split_data(dataFrame)

plt.figure()
plt.scatter(X[:, 0], Y)
plt.title("BID_DA x Damage")
plt.xlabel("BID_DA")
plt.ylabel("Damage")

plt.figure()
plt.scatter(X[:, 1], Y)
plt.title("Road System Type x Damage")
plt.xlabel("Road System Type")
plt.ylabel("Damage")

plt.figure()
plt.scatter(X[:, 2], Y)
plt.title("Total Bid Amount x Damage")
plt.xlabel("Total Bid Amount")
plt.ylabel("Damage")

plt.figure()
plt.scatter(X[:, 3], Y)
plt.title("Net Change Order Amount x Damage")
plt.xlabel("Net Change Order Amount")
plt.ylabel("Damage")

plt.figure()
plt.scatter(X[:, 4], Y)
plt.title("Pending Change Order Amount x Damage")
plt.xlabel("Pending Change Order Amount")
plt.ylabel("Damage")

plt.figure()
plt.scatter(X[:, 5], Y)
plt.title("Auto Liquidated Damage Indicator x Damage")
plt.xlabel("Auto Liquidated Damage Indicator")
plt.ylabel("Damage")

plt.figure()
plt.scatter(X[:, 6], Y)
plt.title("Funding Indicator x Damage")
plt.xlabel("Funding Indicator")
plt.ylabel("Damage")

plt.figure()
plt.scatter(X[:, 7], Y)
plt.title("Total Adjustment Days x Damage")
plt.xlabel("Total Adjustment Days")
plt.ylabel("Damage")

plt.show()
