import os
import subprocess
import signal
import time
import smtplib
from email.message import EmailMessage

print("Starting training for all species")
data_root = 'classified_data'
species_dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]

def terminate_gracefully(process):
    """Attempt graceful termination with timeout"""
    process.send_signal(signal.SIGINT)
    try:
        # Wait up to 5 minutes for process to save and exit
        process.wait(timeout=30)
        return True
    except subprocess.TimeoutExpired:
        # Force kill if graceful shutdown fails
        process.kill()
        return False

def send_email_notification(subject, body):

    """Send email notification using Gmail SMTP"""
    EMAIL_ADDRESS = os.getenv('GMAIL_USER')  

    EMAIL_PASSWORD = os.getenv('GMAIL_APP_PASSWORD')
    
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = EMAIL_ADDRESS

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
    except Exception as e:
        print(f"Failed to send email: {e}")

for species in species_dirs:
    species_path = os.path.join(data_root, species)
    if species not in ["blabul1", "livbul1", "grytre1", "bladro1", "swiwhe1", "eutspa", "taibar2"]:
        continue
    
    command = [
        'python',
        'train.py',
        'exp=base_medium',
        'trainer.gpus=1',
        f'+datamodule.dataset.path={species_path}',
        f'+dataset_name={species}'
    ]
    trained = False
    print(f"Starting training for species: {species}")
    
    try:
        process = subprocess.Popen(command, 
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 universal_newlines=True)
        stdout, stderr = process.communicate(timeout=8*60*60)  # 12 hours
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command, stderr)
            
    except subprocess.TimeoutExpired:
        print(f"Training timeout reached for {species}")
        terminate_gracefully(process)
        trained = True
        
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
        terminate_gracefully(process)
        trained = True
        raise
        
    except (subprocess.SubprocessError, subprocess.CalledProcessError) as e:
        error_msg = f"Process error while training {species}: {str(e)}\nOutput: {stderr}"
        print(error_msg)
        send_email_notification(f"Training Error: {species}", error_msg)
        
    except Exception as e:
        error_msg = f"Unexpected error training {species}: {str(e)}"
        print(error_msg)
        send_email_notification(f"Fatal Error: {species}", error_msg)
    
    finally:
        if process and process.poll() is None:
            process.kill()
        time.sleep(15)
        if trained:
            print(f"Training completed for species: {species}")
        else:
            error_msg = f"Training failed for species: {species}: {stderr}, {stdout}"
            print(error_msg)
            send_email_notification(f"Training Failed: {species}", error_msg)