# WhatsApp Business API - Official Implementation Guide
# This is the LEGITIMATE way to build WhatsApp bots for business use

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WhatsAppConfig:
    """Configuration for WhatsApp Business API"""
    access_token: str
    phone_number_id: str  # Your WhatsApp Business phone number ID
    business_account_id: str  # Your WhatsApp Business Account ID
    webhook_verify_token: str  # Token for webhook verification
    app_id: str  # Meta App ID
    app_secret: str  # Meta App Secret
    api_version: str = "v19.0"  # Current API version
    base_url: str = "https://graph.facebook.com"

class WhatsAppBusinessAPI:
    """
    Official WhatsApp Business API Client
    
    Features:
    - Send messages (text, media, templates)
    - Receive webhooks for incoming messages
    - Manage message templates
    - Handle business conversations
    """
    
    def __init__(self, config: WhatsAppConfig):
        self.config = config
        self.base_url = f"{config.base_url}/{config.api_version}"
        self.headers = {
            "Authorization": f"Bearer {config.access_token}",
            "Content-Type": "application/json"
        }
    
    # SENDING MESSAGES
    def send_text_message(self, to: str, message: str) -> Dict:
        """
        Send a text message to a WhatsApp user
        
        Args:
            to: Recipient's phone number (with country code, no + symbol)
            message: Text message to send
            
        Returns:
            API response
        """
        url = f"{self.base_url}/{self.config.phone_number_id}/messages"
        
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {
                "preview_url": False,
                "body": message
            }
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Message sent successfully: {result}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending message: {e}")
            return {"error": str(e)}
    
    def send_template_message(self, to: str, template_name: str, 
                            template_params: List[str] = None, 
                            language_code: str = "en") -> Dict:
        """
        Send a template message (required for business-initiated conversations)
        
        Args:
            to: Recipient's phone number
            template_name: Name of approved template
            template_params: Parameters for template placeholders
            language_code: Template language code
            
        Returns:
            API response
        """
        url = f"{self.base_url}/{self.config.phone_number_id}/messages"
        
        # Build template components
        components = []
        if template_params:
            parameters = [{"type": "text", "text": param} for param in template_params]
            components.append({
                "type": "body",
                "parameters": parameters
            })
        
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "template",
            "template": {
                "name": template_name,
                "language": {
                    "code": language_code
                },
                "components": components
            }
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Template message sent: {result}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending template: {e}")
            return {"error": str(e)}
    
    def send_media_message(self, to: str, media_type: str, 
                          media_id: str = None, media_url: str = None,
                          caption: str = None) -> Dict:
        """
        Send media message (image, document, audio, video)
        
        Args:
            to: Recipient's phone number
            media_type: Type of media (image, document, audio, video)
            media_id: Media ID (if uploaded to WhatsApp)
            media_url: Direct media URL (if hosting externally)
            caption: Optional caption for media
            
        Returns:
            API response
        """
        url = f"{self.base_url}/{self.config.phone_number_id}/messages"
        
        # Build media object
        media_obj = {}
        if media_id:
            media_obj["id"] = media_id
        elif media_url:
            media_obj["link"] = media_url
        
        if caption and media_type in ["image", "document", "video"]:
            media_obj["caption"] = caption
        
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": media_type,
            media_type: media_obj
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Media message sent: {result}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending media: {e}")
            return {"error": str(e)}
    
    def send_interactive_message(self, to: str, message_type: str, 
                               header: str, body: str, footer: str,
                               buttons: List[Dict] = None, 
                               sections: List[Dict] = None) -> Dict:
        """
        Send interactive message with buttons or list
        
        Args:
            to: Recipient's phone number
            message_type: "button" or "list"
            header: Message header text
            body: Message body text
            footer: Message footer text
            buttons: List of button objects (for button type)
            sections: List of section objects (for list type)
            
        Returns:
            API response
        """
        url = f"{self.base_url}/{self.config.phone_number_id}/messages"
        
        interactive_obj = {
            "type": message_type,
            "header": {"type": "text", "text": header} if header else None,
            "body": {"text": body},
            "footer": {"text": footer} if footer else None
        }
        
        if message_type == "button" and buttons:
            interactive_obj["action"] = {"buttons": buttons}
        elif message_type == "list" and sections:
            interactive_obj["action"] = {"sections": sections}
        
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual", 
            "to": to,
            "type": "interactive",
            "interactive": interactive_obj
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Interactive message sent: {result}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending interactive message: {e}")
            return {"error": str(e)}
    
    # RECEIVING MESSAGES (Webhook Handler)
    def process_webhook(self, webhook_data: Dict) -> List[Dict]:
        """
        Process incoming webhook data from WhatsApp
        
        Args:
            webhook_data: Raw webhook data from Meta
            
        Returns:
            List of processed messages
        """
        messages = []
        
        try:
            if "entry" in webhook_data:
                for entry in webhook_data["entry"]:
                    if "changes" in entry:
                        for change in entry["changes"]:
                            if change.get("field") == "messages":
                                value = change.get("value", {})
                                
                                # Process incoming messages
                                if "messages" in value:
                                    for message in value["messages"]:
                                        processed_msg = self._process_message(message, value)
                                        messages.append(processed_msg)
                                
                                # Process message status updates
                                if "statuses" in value:
                                    for status in value["statuses"]:
                                        processed_status = self._process_status(status)
                                        messages.append(processed_status)
        
        except Exception as e:
            logger.error(f"Error processing webhook: {e}")
        
        return messages
    
    def _process_message(self, message: Dict, value: Dict) -> Dict:
        """Process individual incoming message"""
        sender = message.get("from")
        message_id = message.get("id")
        timestamp = message.get("timestamp")
        message_type = message.get("type")
        
        # Extract contact info
        contacts = value.get("contacts", [])
        sender_name = "Unknown"
        if contacts:
            contact = contacts[0]
            sender_name = contact.get("profile", {}).get("name", sender)
        
        # Extract message content based on type
        content = ""
        if message_type == "text":
            content = message.get("text", {}).get("body", "")
        elif message_type == "image":
            content = f"[Image] {message.get('image', {}).get('caption', '')}"
        elif message_type == "document":
            content = f"[Document] {message.get('document', {}).get('filename', '')}"
        elif message_type == "audio":
            content = "[Audio Message]"
        elif message_type == "video":
            content = f"[Video] {message.get('video', {}).get('caption', '')}"
        elif message_type == "button":
            content = f"[Button Pressed] {message.get('button', {}).get('text', '')}"
        elif message_type == "interactive":
            interactive = message.get("interactive", {})
            if interactive.get("type") == "button_reply":
                content = f"[Button] {interactive.get('button_reply', {}).get('title', '')}"
            elif interactive.get("type") == "list_reply":
                content = f"[List] {interactive.get('list_reply', {}).get('title', '')}"
        
        return {
            "message_id": message_id,
            "sender": sender,
            "sender_name": sender_name,
            "timestamp": datetime.fromtimestamp(int(timestamp)),
            "type": message_type,
            "content": content,
            "raw_message": message
        }
    
    def _process_status(self, status: Dict) -> Dict:
        """Process message status update"""
        return {
            "type": "status",
            "message_id": status.get("id"),
            "status": status.get("status"),  # sent, delivered, read, failed
            "timestamp": datetime.fromtimestamp(int(status.get("timestamp", 0))),
            "recipient": status.get("recipient_id")
        }
    
    # TEMPLATE MANAGEMENT
    def create_message_template(self, template_name: str, category: str,
                              language: str, components: List[Dict]) -> Dict:
        """
        Create a new message template
        
        Args:
            template_name: Unique template name
            category: Template category (MARKETING, UTILITY, AUTHENTICATION)
            language: Language code (e.g., 'en', 'es')
            components: Template components (header, body, footer, buttons)
            
        Returns:
            API response
        """
        url = f"{self.base_url}/{self.config.business_account_id}/message_templates"
        
        payload = {
            "name": template_name,
            "category": category,
            "language": language,
            "components": components
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Template created: {result}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error creating template: {e}")
            return {"error": str(e)}
    
    def get_message_templates(self) -> Dict:
        """Get all message templates for the business account"""
        url = f"{self.base_url}/{self.config.business_account_id}/message_templates"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            result = response.json()
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching templates: {e}")
            return {"error": str(e)}
    
    # UTILITY METHODS
    def mark_message_as_read(self, message_id: str) -> Dict:
        """Mark a message as read"""
        url = f"{self.base_url}/{self.config.phone_number_id}/messages"
        
        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error marking message as read: {e}")
            return {"error": str(e)}


# WEBHOOK SERVER (Flask/FastAPI example)
from flask import Flask, request, jsonify
import hmac
import hashlib

class WhatsAppWebhookServer:
    """
    Flask server to handle WhatsApp webhooks
    This receives incoming messages and processes them
    """
    
    def __init__(self, whatsapp_api: WhatsAppBusinessAPI, config: WhatsAppConfig):
        self.app = Flask(__name__)
        self.whatsapp_api = whatsapp_api
        self.config = config
        self.setup_routes()
        
        # Store for conversation state management
        self.conversations = {}
    
    def setup_routes(self):
        """Setup Flask routes for webhook handling"""
        
        @self.app.route('/webhook', methods=['GET'])
        def verify_webhook():
            """Verify webhook URL during setup"""
            mode = request.args.get('hub.mode')
            token = request.args.get('hub.verify_token')
            challenge = request.args.get('hub.challenge')
            
            if mode == "subscribe" and token == self.config.webhook_verify_token:
                logger.info("Webhook verified successfully")
                return challenge, 200
            else:
                logger.warning("Webhook verification failed")
                return "Verification failed", 403
        
        @self.app.route('/webhook', methods=['POST'])
        def handle_webhook():
            """Handle incoming webhook data"""
            try:
                # Verify webhook signature
                signature = request.headers.get('X-Hub-Signature-256')
                if not self.verify_webhook_signature(request.data, signature):
                    return "Invalid signature", 403
                
                # Process webhook data
                webhook_data = request.json
                messages = self.whatsapp_api.process_webhook(webhook_data)
                
                # Handle each message
                for message in messages:
                    if message.get("type") != "status":  # Don't respond to status updates
                        self.handle_incoming_message(message)
                
                return jsonify({"status": "success"}), 200
                
            except Exception as e:
                logger.error(f"Error handling webhook: {e}")
                return jsonify({"error": str(e)}), 500
    
    def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """Verify webhook signature for security"""
        if not signature:
            return False
        
        expected_signature = hmac.new(
            self.config.app_secret.encode('utf-8'),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        received_signature = signature.replace('sha256=', '')
        return hmac.compare_digest(expected_signature, received_signature)
    
    def handle_incoming_message(self, message: Dict):
        """
        Handle incoming messages with custom business logic
        This is where you implement your bot's functionality
        """
        sender = message["sender"]
        content = message["content"].lower()
        
        logger.info(f"Received message from {message['sender_name']}: {content}")
        
        # Mark message as read
        self.whatsapp_api.mark_message_as_read(message["message_id"])
        
        # Example bot responses
        if "hello" in content or "hi" in content:
            response = "Hello! Welcome to our WhatsApp Business service. How can I help you today?"
            self.whatsapp_api.send_text_message(sender, response)
        
        elif "help" in content:
            response = """
Here are the commands you can use:
• Type 'info' for business information
• Type 'support' to contact customer support
• Type 'hours' for our business hours
• Type 'location' for our address
            """
            self.whatsapp_api.send_text_message(sender, response)
        
        elif "info" in content:
            # Send business info template
            self.whatsapp_api.send_template_message(
                sender, 
                "business_info_template",  # You need to create this template
                language_code="en"
            )
        
        elif "support" in content:
            # Send interactive message with support options
            buttons = [
                {"type": "reply", "reply": {"id": "tech_support", "title": "Technical Support"}},
                {"type": "reply", "reply": {"id": "billing", "title": "Billing"}},
                {"type": "reply", "reply": {"id": "general", "title": "General Inquiry"}}
            ]
            
            self.whatsapp_api.send_interactive_message(
                sender,
                "button",
                "Support Center",
                "What type of support do you need?",
                "Our team will assist you shortly.",
                buttons=buttons
            )
        
        else:
            # Default response
            response = "Thank you for your message. Our team will get back to you soon. Type 'help' for available commands."
            self.whatsapp_api.send_text_message(sender, response)
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the webhook server"""
        logger.info(f"Starting WhatsApp webhook server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


# EXAMPLE USAGE AND SETUP
def main():
    """Example usage of WhatsApp Business API"""
    
    # Configuration (replace with your actual values)
    config = WhatsAppConfig(
        access_token="YOUR_ACCESS_TOKEN",  # From Meta Business Manager
        phone_number_id="YOUR_PHONE_NUMBER_ID",  # From WhatsApp Business Account
        business_account_id="YOUR_BUSINESS_ACCOUNT_ID",  # From WhatsApp Business Account
        webhook_verify_token="YOUR_WEBHOOK_VERIFY_TOKEN",  # Random string you choose
        app_id="YOUR_APP_ID",  # From Meta App Dashboard
        app_secret="YOUR_APP_SECRET"  # From Meta App Dashboard
    )
    
    # Initialize API client
    whatsapp_api = WhatsAppBusinessAPI(config)
    
    # Example: Send a text message
    result = whatsapp_api.send_text_message(
        to="1234567890",  # Recipient's phone number
        message="Hello from WhatsApp Business API!"
    )
    print(f"Message sent: {result}")
    
    # Example: Send a template message
    template_result = whatsapp_api.send_template_message(
        to="1234567890",
        template_name="hello_world",  # Default template provided by Meta
        language_code="en_US"
    )
    print(f"Template sent: {template_result}")
    
    # Start webhook server for receiving messages
    webhook_server = WhatsAppWebhookServer(whatsapp_api, config)
    webhook_server.run(debug=True)


if __name__ == "__main__":
    main()

# SETUP INSTRUCTIONS:
"""
STEP-BY-STEP SETUP GUIDE:

1. PREREQUISITES:
   - Business website or app
   - Facebook Business Manager account
   - Valid business documentation
   - Dedicated phone number (not used on personal WhatsApp)

2. CREATE META APP:
   - Go to developers.facebook.com
   - Create new app → Business → Business
   - Add WhatsApp product to your app

3. GET ACCESS TOKENS:
   - In App Dashboard → WhatsApp → API Setup
   - Add phone number and verify it
   - Generate temporary access token (24h) for testing
   - Generate permanent access token for production

4. SETUP WEBHOOK:
   - Deploy this code to a server with HTTPS
   - Configure webhook URL in Meta App Dashboard
   - Verify webhook with your verify token

5. CREATE MESSAGE TEMPLATES:
   - All business-initiated messages must use approved templates
   - Create templates in Meta Business Manager
   - Wait for approval (usually 24-48 hours)

6. TESTING:
   - Use Graph API Explorer for initial testing
   - Test with your own phone number first
   - Gradually add more users

7. GO LIVE:
   - Submit app for review by Meta
   - Provide business verification documents
   - Wait for approval (1-2 weeks)

PRICING (as of 2025):
- Free tier: 1,000 conversations/month
- Paid: $0.005 - $0.15 per conversation (varies by country)
- Template message pricing starting July 2025

IMPORTANT NOTES:
- 24-hour rule: Can only send free-form messages within 24h of user's last message
- Template messages can be sent anytime but require approval
- Rate limits apply (80 messages per second)
- Business account required, not personal account
"""