# Unofficial Implementation of the Titan Architecture by Google Research

## Description
An attemtp to implement and understand Google's Titan architecture in the field of Continual Learning

So far, a memory Module Prototype has been implemented

A Transformer using the persistent memory architecture described in the paper has been added.

Next step: integrating the Persistent Transformer with the Memory Module in the three ways described in the paper

## Model Architecture

Titans are a modification of the conventional Transformer architecture. They add a Memory Module that stores data in it's neural weights, while the overall Transformer weights remain fixed.

In the paper and in this implementation, a simple MLP was used.

<!---## Installation

## Usage

## License
MIT(?)
-->