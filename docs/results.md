# Results

Written by `uv run python run.py report` on 05 October 2026, from the saved files in `data/`. Making this page called no LLM and cost nothing.

## Chapter 2: the LLM teaches the ML model

- The LLM labelled **1,080** messages. Its labels match the dataset's true labels **91.5%** of the time.

Where the LLM disagreed most (often a close pair, not a real mistake):

| Dataset says | LLM said | Messages |
|---|---|---|
| check_invoice | get_invoice | 16 |
| track_refund | get_refund | 12 |
| delivery_options | place_order | 8 |
| delete_account | cancel_order | 6 |
| edit_account | change_order | 5 |
| set_up_shipping_address | change_shipping_address | 4 |

The receptionist (the ML model) on 540 clean test messages: **92.4%** right. On 50 messy messages: **58.0%**.

| Answers only when at least | Share of messages it answers | Right on those |
|---|---|---|
| 40% sure | 96% | 92.6% |
| 60% sure | 86% | 95.7% |
| 80% sure | 72% | 98.3% |
| 90% sure | 52% | 98.7% |

## Chapter 3: expert only vs the whole office

60 messages (a mix of normal and messy ones), each sent to the senior expert alone and to the whole office.

|  | Expert only | Whole office |
|---|---|---|
| Total cost | $0.059 | $0.031 |
| Cost per 1,000 messages | $0.99 | $0.52 |
| Average time per message | 1.70 s | 1.04 s |
| Answered for $0 | 0% | 55% |

**The whole office was 1.9× cheaper and 1.6× faster.**

| Helper | Messages | Average time | Average cost |
|---|---|---|---|
| Senior expert (cloud LLM agent) | 27 | 2.00 s | $0.0011 |
| Junior assistant (local LLM) | 7 | 1.21 s | $0.0000 |
| Receptionist (ML model) | 26 | 0.00 s | $0.0000 |

### Every message, both answers

| Message | Who answered | Expert only | Whole office |
|---|---|---|---|
| how can I send feedback about a service? | Receptionist (ML model) | You can share feedback by replying here with the details, and I’ll pass it along to the team. If you want, tell me what happened and whether it was about delive | Thank you! You can leave a review on the product page under My Orders. |
| I'd like to unsubscrie to the corporate newsletter help me | Receptionist (ML model) | Sure — I can help with that. Please share the email address used for the newsletter or your order number if you have one, and I’ll check the right next step. | You can subscribe or unsubscribe under Account → Settings → Emails. |
| wheres my stuf, order no NM10101 | Senior expert (cloud LLM agent) | Your order NM10101 is marked as cancelled, so it won’t be delivered. If you want, I can help you with the next step or you can share if you expected a refund is | Your order NM10101 is marked as cancelled, so it won’t be delivered. If you want, I can help you with the next step or you can share if you expected a refund is |
| want help to see ur reimbursement policy | Receptionist (ML model) | Sure — NovaMart refunds can be requested within **30 days of delivery** for orders with status **Delivered**. The refund is sent back to the **original payment  | You can ask for a refund within 30 days of delivery. The money is back in 5-7 working days. |
| chnge data on pro account | Junior assistant (local LLM) | Sure — I can help with account changes, but I need a bit more detail. Please tell me what you want to change on your Pro account (for example: name, email, phon | To change data on your Pro account, follow these steps:  1. Sign in to your NovaMart account. 2. Go to **Account → Settings**. 3. Click on **Change Data**. 4. U |
| what payment methods do u accept | Receptionist (ML model) | We accept UPI, credit cards, debit cards, net banking, and cash on delivery for orders up to ₹5,000. If you want, I can also help with anything else about check | We accept UPI, credit and debit cards, net banking, and cash on delivery for orders up to ₹5,000. |
| i expect a refund of ₹1,499 dollars | Senior expert (cloud LLM agent) | I can help with a refund, but I need your order number first. Please send the NovaMart order ID (for example, NM10234), and I’ll check whether it’s eligible. | I can help with a refund, but I need your order number first. Please send the NovaMart order ID (for example, NM10234), and I’ll check whether it’s eligible. |
| what do i have to do to list ur allowed payment options | Receptionist (ML model) | NovaMart accepts UPI, credit cards, debit cards, net banking, and cash on delivery for orders up to ₹5,000. If you want, I can also help with which payment opti | We accept UPI, credit and debit cards, net banking, and cash on delivery for orders up to ₹5,000. |
| I need assistance to get a reimbursement | Senior expert (cloud LLM agent) | Please share your order number (for example, NM10234). I need to look it up first before I can help with a reimbursement. Please share your order number (for ex | Please share your order number (for example, NM10234). I need to look it up first before I can help with a reimbursement. Please share your order number (for ex |
| help checking what hours i can reach customer support | Receptionist (ML model) | You can reach NovaMart support 24/7 via chat here. You can also email support@novamart.example anytime. | You can chat with us here 24/7 or email support@novamart.example. |
| I have chosen a wrong shipping address, how do I correct it? | Senior expert (cloud LLM agent) | If your order is still Processing, I can help update the shipping address for you. Please send me your order number and the new full address. | If your order is still Processing, I can help update the shipping address for you. Please send me your order number and the new full address. |
| what delivery options r there | Receptionist (ML model) | We offer two delivery options: Standard delivery (3–5 working days, free on orders above ₹499) and Express delivery (1–2 working days, ₹99 extra). We deliver to | Standard delivery takes 3-5 working days (free above ₹499). Express takes 1-2 days for ₹99. |
| I want assistance placing an order | Junior assistant (local LLM) | Sure — I can help with that. Please tell me what you want to buy, and if you already have a NovaMart account or shipping details ready. | To place an order, you'll need to sign in to your account or create one if you don't have one. Then, select the items you'd like to purchase from our website an |
| i dont know how i can talk with someone | Senior expert (cloud LLM agent) | I’ve passed this to a human agent for you. You should get a reply within 2 hours. | I’ve passed this to a human agent for you. You should get a reply within 2 hours. |
| where do I notify of an error with signup? | Receptionist (ML model) | I can help with that. If you’re having a signup error, please tell me the exact message you see and I’ll guide you; if you want, I can also connect you to a hum | Sorry about that! Please try again on the Sign in page, or tell me the error you see. |
| hlw do I add several products to purchase NM10112? | Senior expert (cloud LLM agent) | Hi! I checked order NM10112 and it’s already shipped, so you can’t add more products to this purchase now. If you want, I can help with a new order or check whe | Hi! I checked order NM10112 and it’s already shipped, so you can’t add more products to this purchase now. If you want, I can help with a new order or check whe |
| where do i check the termination penalty | Receptionist (ML model) | I couldn’t find any “termination penalty” in NovaMart’s policy. If you mean account deletion, there’s no fee mentioned, but a human agent must confirm the delet | There is no cancellation fee. Orders can be cancelled for free while they are Processing. |
| i dont know how i can inform of sign-up problems | Receptionist (ML model) | I can help with that. If you’re having sign-up issues, please tell me what happens when you try to register and, if possible, share any error message you see. I | Sorry about that! Please try again on the Sign in page, or tell me the error you see. |
| need help filing a claim against your organization | Senior expert (cloud LLM agent) | I can help with that, but I’ll need your order number first so I can look up the case. If you’re asking for a formal complaint or claim about something beyond a | I can help with that, but I’ll need your order number first so I can look up the case. If you’re asking for a formal complaint or claim about something beyond a |
| can you give me information about a pro account termination? | Senior expert (cloud LLM agent) | For a Pro account termination, a human agent needs to confirm it. I can connect you now—just say the word, and I’ll hand it over. | For a Pro account termination, a human agent needs to confirm it. I can connect you now—just say the word, and I’ll hand it over. |
| whats the status of my refund for NM10050 | Senior expert (cloud LLM agent) | I checked NM10050, and it’s currently marked **Delayed** with delivery expected on **10 Oct 2026**. I don’t see a refund started yet for this order—if you want, | I checked NM10050, and it’s currently marked **Delayed** with delivery expected on **10 Oct 2026**. I don’t see a refund started yet for this order—if you want, |
| i need help to delete something from order NM10234 | Senior expert (cloud LLM agent) | I can help with that, but I need to know what you want to delete from order NM10234. Since it’s still Processing, items or the delivery address can be changed;  | I can help with that, but I need to know what you want to delete from order NM10234. Since it’s still Processing, items or the delivery address can be changed;  |
| reset pasword plz | Receptionist (ML model) | I can help, but I’ll need a bit more info. If you mean your NovaMart account, I can guide you, but I can’t reset passwords directly here—please use the “Forgot  | Click Sign in, then 'Forgot password'. We will email you a reset link. |
| how do i make an account | Senior expert (cloud LLM agent) | You can create a NovaMart account from the app or website by tapping **Sign Up / Create Account** and entering your phone number or email, then verifying the OT | You can create a NovaMart account from the app or website by tapping **Sign Up / Create Account** and entering your phone number or email, then verifying the OT |
| I have to see how soon can I expect my product, help me | Receptionist (ML model) | Sure — please share your order number (for example, NM10234), and I’ll check the delivery status for you. | Standard delivery takes 3-5 working days. Express delivery takes 1-2 working days. |
| i want to add one more item to order NM10003 | Senior expert (cloud LLM agent) | Order NM10003 is already delivered, so I can’t add another item to it. If you’d like, I can help you with a refund for the delivered order or guide you on placi | Order NM10003 is already delivered, so I can’t add another item to it. If you’d like, I can help you with a refund for the delivered order or guide you on placi |
| i moved!! send NM10199 to my new flat in Andheri Mumbai | Senior expert (cloud LLM agent) | I checked order NM10199, and it’s already cancelled, so I can’t change the delivery address. If you want, I can help with a new order instead. | I checked order NM10199, and it’s already cancelled, so I can’t change the delivery address. If you want, I can help with a new order instead. |
| information about cancel the sbscription to the newsletter | Receptionist (ML model) | I can help with that, but I need to check NovaMart’s policy first. Please share your order number if this is linked to an order, or I can guide you on general n | You can subscribe or unsubscribe under Account → Settings → Emails. |
| how many days do i have to return something | Junior assistant (local LLM) | You can request a refund within 30 days of delivery. The order also needs to be marked as Delivered. | You can return an item within **30 days of delivery**. |
| help seeing the termination charges | Receptionist (ML model) | NovaMart doesn’t charge any termination or cancellation fees. If you meant cancelling an order, it’s free while the order is still Processing. | There is no cancellation fee. Orders can be cancelled for free while they are Processing. |
| open a platinum account for my dad | Junior assistant (local LLM) | I can help with account requests, but I’ll need a human agent to handle opening a Platinum account. Please share the account holder’s details, and I’ll escalate | To open a platinum account for your dad, please follow these steps:  1. Go to the NovaMart website and sign in to your account. 2. Click on "Account" and then " |
| how can I remove some articles from purchase NM10035? | Senior expert (cloud LLM agent) | I checked order NM10035, and it has already been delivered, so the items can’t be changed or removed now. If you want, I can start a refund request for the orde | I checked order NM10035, and it has already been delivered, so the items can’t be changed or removed now. If you want, I can start a refund request for the orde |
| let me talk to a real person not a bot | Senior expert (cloud LLM agent) | Of course — I’ve connected you to a human agent. You should hear back within 2 hours. | Of course — I’ve connected you to a human agent. You should hear back within 2 hours. |
| need assistance seeing the withdrawal penalty | Receptionist (ML model) | I can help with refunds/cancellations, but I don’t see any “withdrawal penalty” in NovaMart’s policy. If you mean a refund or cancellation fee, there isn’t one, | There is no cancellation fee. Orders can be cancelled for free while they are Processing. |
| need help seeing the eta of the purchase NM10125 | Senior expert (cloud LLM agent) | Your order NM10125 is delayed, and the current delivery date is 12 Oct 2026. If you want, I can also help check whether anything else is needed from your side. | Your order NM10125 is delayed, and the current delivery date is 12 Oct 2026. If you want, I can also help check whether anything else is needed from your side. |
| do you take upi? | Senior expert (cloud LLM agent) | Yes — we accept UPI, along with credit/debit cards, net banking, and cash on delivery for orders up to ₹5,000. | Yes — we accept UPI, along with credit/debit cards, net banking, and cash on delivery for orders up to ₹5,000. |
| have a questoon about creating user profiles | Junior assistant (local LLM) | I can help with that. What would you like to know about creating user profiles? | How do I create a user profile on NovaMart? |
| delete premium acvount | Senior expert (cloud LLM agent) | I’ve passed this to a human agent for account deletion. You should get a reply within 2 hours. | I’ve passed this to a human agent for account deletion. You should get a reply within 2 hours. |
| can ya show me at what time customr service available is | Receptionist (ML model) | We’re available 24/7 on chat here, and you can also email support@novamart.example anytime. | You can chat with us here 24/7 or email support@novamart.example. |
| help to cancel purchase NM10108 | Senior expert (cloud LLM agent) | I checked NM10108, and it’s already **Shipped**, so I can’t cancel it now. If you want, I can help you request a refund after delivery. | I checked NM10108, and it’s already **Shipped**, so I can’t cancel it now. If you want, I can help you request a refund after delivery. |
| cancel NM10019 asap i ordered by mistake | Senior expert (cloud LLM agent) | Your order NM10019 is already cancelled, so there’s nothing more to do. If you need help with anything else, just let me know. | Your order NM10019 is already cancelled, so there’s nothing more to do. If you need help with anything else, just let me know. |
| cqn i swap several items of purchase NM10009 | Senior expert (cloud LLM agent) | I checked order NM10009, and it’s already shipped, so we can’t swap or change items on this order. If you want, I can help you with the next best option after d | I checked order NM10009, and it’s already shipped, so we can’t swap or change items on this order. If you want, I can help you with the next best option after d |
| forgot my password cant log in | Junior assistant (local LLM) | I can help, but I can’t reset passwords here. Please use the “Forgot Password” option on the login page, or if you’re having trouble with account access I can c | To reset your password, go to the **Sign in** page and click on **Forgot Password**. Enter your email address or phone number associated with your account, and  |
| plz cancel my ordr NM10152 | Senior expert (cloud LLM agent) | Sorry, I can’t cancel NM10152 because it has already been delivered. If you’d like, I can help you start a refund instead. | Sorry, I can’t cancel NM10152 because it has already been delivered. If you’d like, I can help you start a refund instead. |
| what do i need to do to check the status of the restitution | Receptionist (ML model) | Please share your order number (for example, NM10234), and I’ll check the status for you. | Sure! What is your order number? It looks like NM10234. |
| this is a scam, never buying from novamart again | Senior expert (cloud LLM agent) | I’m sorry for the bad experience. I’ve escalated this to a human agent, and they’ll reply within 2 hours. | I’m sorry for the bad experience. I’ve escalated this to a human agent, and they’ll reply within 2 hours. |
| bhai mera order NM10045 kab aayega | Senior expert (cloud LLM agent) | Bhai, aapka order NM10045 delayed hai aur abhi estimated delivery date 09 Oct 2026 dikh rahi hai. Agar chaho to main aur details check karne mein help kar sakta | Bhai, aapka order NM10045 delayed hai aur abhi estimated delivery date 09 Oct 2026 dikh rahi hai. Agar chaho to main aur details check karne mein help kar sakta |
| where can i download invoice | Receptionist (ML model) | I can help with that, but I need your order number first to check the invoice availability. Please share your NovaMart order ID (for example, NM10234). | You can download your invoice from My Orders on the website. It is also emailed when the order ships. |
| download invoices from Rahul | Receptionist (ML model) | Please share the order number for Rahul’s order, and I can help check the invoice. | You can download your invoice from My Orders on the website. It is also emailed when the order ships. |
| payment failed but money got deducted from my account | Senior expert (cloud LLM agent) | I’m sorry about that — this needs a human agent since it’s a payment issue. I’ve escalated it, and a support rep should reply within 2 hours. | I’m sorry about that — this needs a human agent since it’s a payment issue. I’ve escalated it, and a support rep should reply within 2 hours. |
| I want assistance to set the goddamn delivery address up | Receptionist (ML model) | Sure — I can help update it, but I need your order number first. Please send it in the format **NM12345** and your new delivery address. Sure — I can help updat | You can add a shipping address under Account → Addresses, or at checkout. |
| i dont know how i can see ur reimbursement policy | Receptionist (ML model) | NovaMart’s refund policy is: you can request a refund within 30 days of delivery, only for orders marked Delivered. Refunds go back to the original payment meth | You can ask for a refund within 30 days of delivery. The money is back in 5-7 working days. |
| help me check what hours i can call customer service | Receptionist (ML model) | You can chat with NovaMart support 24/7 here, or email support@novamart.example. If you want, I can also help with a specific order or issue. | You can chat with us here 24/7 or email support@novamart.example. |
| would it be possible to sign up to the newsletter? | Receptionist (ML model) | Yes — you can usually sign up for the NovaMart newsletter from your account settings or the homepage footer. If you want, I can also help with anything else on  | You can subscribe or unsubscribe under Account → Settings → Emails. |
| can ya help me checking the cancellation charges | Receptionist (ML model) | Sure — NovaMart has **no cancellation fee**. If your order is still **Processing**, you can cancel it for free; if it’s already **Shipped**, it can’t be cancell | There is no cancellation fee. Orders can be cancelled for free while they are Processing. |
| help to leave damn feedback for ru products | Receptionist (ML model) | I can help with that. Please share your order number first, and I’ll check the order details so I can guide you the right way. | Thank you! You can leave a review on the product page under My Orders. |
| your app charged me twice for the same order!!! | Senior expert (cloud LLM agent) | I’m sorry about that—duplicate charges need a human review. I’ve escalated this as high priority, and a human agent will reply within 2 hours. | I’m sorry about that—duplicate charges need a human review. I’ve escalated this as high priority, and a human agent will reply within 2 hours. |
| i dont khow what i need to do to change to the gold account | Junior assistant (local LLM) | I can help with that, but I need a bit more detail first. Are you trying to upgrade your NovaMart account to Gold membership, and do you have an order number or | To upgrade to the Gold account, go to **Account → Upgrade to Gold** on the website. Fill in the required details and submit the request. A human agent will revi |
| I cannot create a user profile, help reporting signup issues | Receptionist (ML model) | I can help with that. Please share the email/phone you used and, if you have it, any error message or screenshot text from the signup page. If this is a payment | Sorry about that! Please try again on the Sign in page, or tell me the error you see. |
| absolutely terrible product, the box was broken and nobody cares | Senior expert (cloud LLM agent) | I’m sorry about that. If your order was delivered, I can help start a refund for the damaged box—please send your order number so I can look it up first. | I’m sorry about that. If your order was delivered, I can help start a refund for the damaged box—please send your order number so I can look it up first. |

## Saved answers

`data/saved_llm_answers.jsonl` holds **128** real LLM answers (models: `gpt-5.4-mini`, `llama3.2:3b`), which cost **$0.07** when they were made. The app, notebooks, chat and benchmark replay them for free whenever the same request comes again.

