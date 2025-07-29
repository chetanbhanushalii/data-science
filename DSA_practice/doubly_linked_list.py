class Node:
    def __init__(self, data=None, next=None, prev=None):
        self.data = data
        self.next = next
        self.prev = prev 

class LinkedList:
    def __init__(self):
        self.head = None

    def print_forward(self):
    # This method prints list in forward direction. Use node.next
        if self.head is None:
            print("Linked list is empty")
            return
        itr = self.head
        llstr = ''
        while itr:
            llstr += str(itr.data)+' --> ' if itr.next else str(itr.data)
            itr = itr.next
        print(llstr)

    def print_backward(self):
    # Print linked list in reverse direction. Use node.prev for this.
        if self.head is None:
            print("Linked list is empty")
            return
        itr = self.head
        while itr.next:
            itr = itr.next
        # Now itr is at the last node
        # We will print the linked list in reverse direction
        # We can use a while loop to traverse backwards using prev pointer
        # We will use a temporary variable to store the previous node
        # and print the data
        llstr = ''
        while itr:
            llstr += str(itr.data)+' --> ' if itr.prev else str(itr.data)
            itr = itr.prev
        print(llstr)
    
    def print(self):
        if self.head is None:
            print("Linked list is empty")
            return
        itr = self.head
        llstr = ''
        while itr:
            llstr += str(itr.data)+' --> ' if itr.next else str(itr.data)
            itr = itr.next
        print(llstr)

    def get_length(self):
        count = 0
        itr = self.head
        while itr:
            count+=1
            itr = itr.next
        return count

    def insert_at_begining(self, data):
        if self.head == None:
            node = Node(data, self.head, None)
            self.head = node
        else:
            node = Node(data, self.head, None)
            self.head.prev = node
            self.head = node

    def insert_at_end(self, data):
        if self.head is None:
            self.head = Node(data, None, None)
            return

        itr = self.head

        while itr.next:
            itr = itr.next

        itr.next = Node(data, None, itr)

    def insert_at(self, index, data):
        if index<0 or index>self.get_length():
            raise Exception("Invalid Index")

        if index==0:
            self.insert_at_begining(data)
            return

        count = 0
        itr = self.head
        while itr:
            if count == index - 1:
                node = Node(data, itr.next, itr)
                itr.next = node
                if node.next:
                    node.next.prev = node
                break

            itr = itr.next
            count += 1

    def remove_at(self, index):
        if index<0 or index>=self.get_length():
            raise Exception("Invalid Index")

        if index==0:
            self.head = self.head.next
            return

        count = 0
        itr = self.head
        while itr:
            if count == index - 1:
                itr.next = itr.next.next
                itr.next.prev = itr
                break

            itr = itr.next
            count+=1

    def insert_values(self, data_list):
        self.head = None
        for data in data_list:
            self.insert_at_end(data)

    def insert_after_value(self, data_after, data_to_insert):
        # Search for first occurance of data_after value in linked list
        # Now insert data_to_insert after data_after node
        itr = self.head
        while itr:
            if itr.data == data_after:
                node = Node(data_to_insert, itr.next, itr)
                itr.next = node
                if node.next:
                    node.next.prev = node
                return
            itr = itr.next
        raise Exception(f"Value {data_after} not found in the list")

    def remove_by_value(self, data):
        itr = self.head
        index = 0
        while itr:
            if itr.data == data:
                self.remove_at(index)
                return
            itr = itr.next
            index += 1
        raise Exception(f"Value {data} not found in the list")

if __name__ == '__main__':
    ll = LinkedList()
    ll.insert_values(["banana","mango","grapes","orange"])
    ll.print_forward()
    ll.print_backward()
    
    # ll.insert_at_begining("apple")
    # ll.print_forward()
    # ll.print_backward()

    ll.insert_at(1,"blueberry")
    ll.print_forward()
    ll.print_backward()

    
    # ll.remove_at(2)
    # ll.print_forward()
    # ll.print_backward()
    
    
    # ll.insert_after_value("blueberry","banana") 
    # ll.print_forward()
    # ll.print_backward()
    
    ll.remove_by_value("mango")
    ll.print_forward()
    ll.print_backward()
