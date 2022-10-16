from distilbert import *

def main():
    model_path= "None"

    label_words = ['*','**','***','****','*****']
    model = DistilBertModelForClassification(label_words=label_words,model_path=model_path)
    
    check = True
    prompt(model)
    
    while check:
        answer = str(input("do you want to review again? (y/n) "))
        if answer == "y":
            prompt(model)

        elif answer == "n":
            print("exiting...")
            check=True
            break
        else:
            print("invalid response, chose (y/n)")

def prompt(model):
    text = str(input("Input: "))
    sentiment = model.infer(text)
    print("Sentiment Review: ",sentiment)


if __name__ == "__main__":
    main()