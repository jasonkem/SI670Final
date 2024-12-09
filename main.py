import pandas as pd
import matplotlib.pyplot as plt

def main():
    try:
        # Read the JSON file
        df = pd.read_json('yelp_photos/photos.json', lines=True)
        
        # Get label distribution
        label_dist = df['label'].value_counts()
        
        # Print numerical distribution
        print("\nPhoto Distribution by Label:")
        print(label_dist)
        print("\nPercentage Distribution:")
        print(label_dist / len(df) * 100, "%")
        
        # Create a bar plot
        plt.figure(figsize=(10, 6))
        label_dist.plot(kind='bar')
        plt.title('Distribution of Yelp Photos by Label')
        plt.xlabel('Label')
        plt.ylabel('Number of Photos')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
