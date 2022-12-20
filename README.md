#Abstract

In the modern world, more and more systems are being created to simplify human interaction with a computer. Systems that parse and understand human language allow more and more people to perform the operations they need. One of the promising technologies is the technology of vector representation of a word. It is known that word vectors have the property of transferring semantics to related words. The idea is to use this property to find semantically close sentences in a text corpus using vector representation of words.

To investigate the effectiveness of this approach, it is necessary to explore various methods for obtaining a vector representation of words and compare them with each other by solving the problem of semantic similarity. Then it is necessary to develop a question-answer system that can, in a special text corpus (FAQ collection), determine the question that is most similar to the user's question. It is important that the question-answer system not only found the most semantically close question from the FAQ collection, but also summed up the answer to the user's question, based on the answer that is paired with the most semantically close question from the FAQ collection.parison of questions only provides a statistical approximation solution, an additional fine tuning has been implemented, which is described in chapter 3.
Since the answers are paired with questions, the right answer will be found as soon as a suitable question is found. One could return this answer without any change to the user, but the answers often are in the form of statements that do not fit. Therefore, an analysis and processing of the answers is also necessary.

The purpose of this work is to study the effectiveness of the word vector representation technology for recognising the semantics of natural language phrases.

# Getting Started with NLP App

To begin using the application it is necessary to add a web plugin to a chrome browser and start a web application.

## Chrome plugin

Chrome plugins are useful additions to already existing features on the website and in the browser. Every Chrome extension must have a manifest.json. It is used for definitions of application features, general description, version numbers, and permissions.

The KSReader.js file specifies additional rules for the plugin, such as which pages the plugin will be active on or which parameters the URL should contain. The user interface is essentially an HTML form (popup.html). The plugin sends a URL to the QA system application every time the user requests something through the plugin interface.

The user enters a question in the input field. After pressing the Enter key, the question and the URL are sent to the server. The plugin has been implemented in such a way that it will only be enabled if the user is on an Amazon product detail page with an ASIN in the URL. The plugin also takes on the task of displaying the response of the QA system.