# Import Packages
import mailbox
import pandas as pd

# Function to Convert Email Data from .mbox to .csv
def convert (address):
    ## Load Email Data
    mbox = mailbox.mbox (address + ".mbox")

    ## Create Empty Email List
    emails_list = []

    ## Extract Email Data
    for message in mbox:
        try:
            ### Extract Email Subject Line
            subject = message ["subject"]

            ### Extract Email Message Body
            body = ""

            if message.is_multipart ():
                for part in message.walk ():
                    if part.get_content_type () == "text/plain":
                        body = part.get_payload (decode = True).decode (errors = "ignore")
                        break
            
            else:
                body = message.get_payload (decode = True).decode (errors = "ignore")
            
            ### Append to the Email List
            emails_list.append ({
                "Subject": subject,
                "Body": body
            })
        
        except Exception as e:
            print (f"Failed to Process Message {e}")
    
    ## Convert Email List to Data Frame
    emails_df = pd.DataFrame (emails_list)

    ## Write to a CSV File
    emails_df.to_csv (address + ".csv", index = False)

convert ("personal_email")
convert ("school_email")
convert ("work_email")
