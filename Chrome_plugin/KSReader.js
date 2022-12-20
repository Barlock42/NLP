'use strict';

chrome.runtime.onInstalled.addListener(function() {
  console.log('KSreader installed.');
  chrome.declarativeContent.onPageChanged.removeRules(undefined, function() {
    chrome.declarativeContent.onPageChanged.addRules([{
      conditions: [new chrome.declarativeContent.PageStateMatcher({
        pageUrl: {hostContains: '.amazon.com', pathContains: 'dp'},
      })],
      actions: [new chrome.declarativeContent.ShowPageAction()]
    }]);
  });
});
